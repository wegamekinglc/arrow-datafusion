// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Expressions

use crate::field_util::get_indexed_field;
use crate::operator::Operator;
use crate::window_frame;
use crate::window_function;
use arrow::{compute::can_cast_types, datatypes::DataType};
use datafusion_common::{
    Column, DFField, DFSchema, DataFusionError, ExprSchema, Result, ScalarValue,
};
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::Not;
use std::sync::Arc;

/// `Expr` is a central struct of DataFusion's query API, and
/// represent logical expressions such as `A + 1`, or `CAST(c1 AS
/// int)`.
///
/// An `Expr` can compute its [DataType](arrow::datatypes::DataType)
/// and nullability, and has functions for building up complex
/// expressions.
///
/// # Examples
///
/// ## Create an expression `c1` referring to column named "c1"
/// ```
/// # use datafusion::logical_plan::*;
/// let expr = col("c1");
/// assert_eq!(expr, Expr::Column(Column::from_name("c1")));
/// ```
///
/// ## Create the expression `c1 + c2` to add columns "c1" and "c2" together
/// ```
/// # use datafusion::logical_plan::*;
/// let expr = col("c1") + col("c2");
///
/// assert!(matches!(expr, Expr::BinaryExpr { ..} ));
/// if let Expr::BinaryExpr { left, right, op } = expr {
///   assert_eq!(*left, col("c1"));
///   assert_eq!(*right, col("c2"));
///   assert_eq!(op, Operator::Plus);
/// }
/// ```
///
/// ## Create expression `c1 = 42` to compare the value in coumn "c1" to the literal value `42`
/// ```
/// # use datafusion::logical_plan::*;
/// # use datafusion::scalar::*;
/// let expr = col("c1").eq(lit(42));
///
/// assert!(matches!(expr, Expr::BinaryExpr { ..} ));
/// if let Expr::BinaryExpr { left, right, op } = expr {
///   assert_eq!(*left, col("c1"));
///   let scalar = ScalarValue::Int32(Some(42));
///   assert_eq!(*right, Expr::Literal(scalar));
///   assert_eq!(op, Operator::Eq);
/// }
/// ```
#[derive(Clone, PartialEq, Hash)]
pub enum Expr {
    /// An expression with a specific name.
    Alias(Box<Expr>, String),
    /// A named reference to a qualified filed in a schema.
    Column(Column),
    /// A named reference to a variable in a registry.
    ScalarVariable(Vec<String>),
    /// A constant value.
    Literal(ScalarValue),
    /// A binary expression such as "age > 21"
    BinaryExpr {
        /// Left-hand side of the expression
        left: Box<Expr>,
        /// The comparison operator
        op: Operator,
        /// Right-hand side of the expression
        right: Box<Expr>,
    },
    /// Negation of an expression. The expression's type must be a boolean to make sense.
    Not(Box<Expr>),
    /// Whether an expression is not Null. This expression is never null.
    IsNotNull(Box<Expr>),
    /// Whether an expression is Null. This expression is never null.
    IsNull(Box<Expr>),
    /// arithmetic negation of an expression, the operand must be of a signed numeric data type
    Negative(Box<Expr>),
    /// Returns the field of a [`ListArray`] or [`StructArray`] by key
    GetIndexedField {
        /// the expression to take the field from
        expr: Box<Expr>,
        /// The name of the field to take
        key: ScalarValue,
    },
    /// Whether an expression is between a given range.
    Between {
        /// The value to compare
        expr: Box<Expr>,
        /// Whether the expression is negated
        negated: bool,
        /// The low end of the range
        low: Box<Expr>,
        /// The high end of the range
        high: Box<Expr>,
    },
    /// The CASE expression is similar to a series of nested if/else and there are two forms that
    /// can be used. The first form consists of a series of boolean "when" expressions with
    /// corresponding "then" expressions, and an optional "else" expression.
    ///
    /// CASE WHEN condition THEN result
    ///      [WHEN ...]
    ///      [ELSE result]
    /// END
    ///
    /// The second form uses a base expression and then a series of "when" clauses that match on a
    /// literal value.
    ///
    /// CASE expression
    ///     WHEN value THEN result
    ///     [WHEN ...]
    ///     [ELSE result]
    /// END
    Case {
        /// Optional base expression that can be compared to literal values in the "when" expressions
        expr: Option<Box<Expr>>,
        /// One or more when/then expressions
        when_then_expr: Vec<(Box<Expr>, Box<Expr>)>,
        /// Optional "else" expression
        else_expr: Option<Box<Expr>>,
    },
    /// Casts the expression to a given type and will return a runtime error if the expression cannot be cast.
    /// This expression is guaranteed to have a fixed type.
    Cast {
        /// The expression being cast
        expr: Box<Expr>,
        /// The `DataType` the expression will yield
        data_type: DataType,
    },
    /// Casts the expression to a given type and will return a null value if the expression cannot be cast.
    /// This expression is guaranteed to have a fixed type.
    TryCast {
        /// The expression being cast
        expr: Box<Expr>,
        /// The `DataType` the expression will yield
        data_type: DataType,
    },
    /// A sort expression, that can be used to sort values.
    Sort {
        /// The expression to sort on
        expr: Box<Expr>,
        /// The direction of the sort
        asc: bool,
        /// Whether to put Nulls before all other data values
        nulls_first: bool,
    },
    /// Represents the call of a built-in scalar function with a set of arguments.
    ScalarFunction {
        /// The function
        fun: functions::BuiltinScalarFunction,
        /// List of expressions to feed to the functions as arguments
        args: Vec<Expr>,
    },
    /// Represents the call of a user-defined scalar function with arguments.
    ScalarUDF {
        /// The function
        fun: Arc<ScalarUDF>,
        /// List of expressions to feed to the functions as arguments
        args: Vec<Expr>,
    },
    /// Represents the call of an aggregate built-in function with arguments.
    AggregateFunction {
        /// Name of the function
        fun: aggregate::AggregateFunction,
        /// List of expressions to feed to the functions as arguments
        args: Vec<Expr>,
        /// Whether this is a DISTINCT aggregation or not
        distinct: bool,
    },
    /// Represents the call of a window function with arguments.
    WindowFunction {
        /// Name of the function
        fun: window_function::WindowFunction,
        /// List of expressions to feed to the functions as arguments
        args: Vec<Expr>,
        /// List of partition by expressions
        partition_by: Vec<Expr>,
        /// List of order by expressions
        order_by: Vec<Expr>,
        /// Window frame
        window_frame: Option<window_frame::WindowFrame>,
    },
    /// aggregate function
    AggregateUDF {
        /// The function
        fun: Arc<AggregateUDF>,
        /// List of expressions to feed to the functions as arguments
        args: Vec<Expr>,
    },
    /// Returns whether the list contains the expr value.
    InList {
        /// The expression to compare
        expr: Box<Expr>,
        /// A list of values to compare against
        list: Vec<Expr>,
        /// Whether the expression is negated
        negated: bool,
    },
    /// Represents a reference to all fields in a schema.
    Wildcard,
}

/// Fixed seed for the hashing so that Ords are consistent across runs
const SEED: ahash::RandomState = ahash::RandomState::with_seeds(0, 0, 0, 0);

impl PartialOrd for Expr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let mut hasher = SEED.build_hasher();
        self.hash(&mut hasher);
        let s = hasher.finish();

        let mut hasher = SEED.build_hasher();
        other.hash(&mut hasher);
        let o = hasher.finish();

        Some(s.cmp(&o))
    }
}

impl Expr {
    /// Returns the [arrow::datatypes::DataType] of the expression
    /// based on [ExprSchema]
    ///
    /// Note: [DFSchema] implements [ExprSchema].
    ///
    /// # Errors
    ///
    /// This function errors when it is not possible to compute its
    /// [arrow::datatypes::DataType].  This happens when e.g. the
    /// expression refers to a column that does not exist in the
    /// schema, or when the expression is incorrectly typed
    /// (e.g. `[utf8] + [bool]`).
    pub fn get_type<S: ExprSchema>(&self, schema: &S) -> Result<DataType> {
        match self {
            Expr::Alias(expr, _) | Expr::Sort { expr, .. } | Expr::Negative(expr) => {
                expr.get_type(schema)
            }
            Expr::Column(c) => Ok(schema.data_type(c)?.clone()),
            Expr::ScalarVariable(_) => Ok(DataType::Utf8),
            Expr::Literal(l) => Ok(l.get_datatype()),
            Expr::Case { when_then_expr, .. } => when_then_expr[0].1.get_type(schema),
            Expr::Cast { data_type, .. } | Expr::TryCast { data_type, .. } => {
                Ok(data_type.clone())
            }
            Expr::ScalarUDF { fun, args } => {
                let data_types = args
                    .iter()
                    .map(|e| e.get_type(schema))
                    .collect::<Result<Vec<_>>>()?;
                Ok((fun.return_type)(&data_types)?.as_ref().clone())
            }
            Expr::ScalarFunction { fun, args } => {
                let data_types = args
                    .iter()
                    .map(|e| e.get_type(schema))
                    .collect::<Result<Vec<_>>>()?;
                functions::return_type(fun, &data_types)
            }
            Expr::WindowFunction { fun, args, .. } => {
                let data_types = args
                    .iter()
                    .map(|e| e.get_type(schema))
                    .collect::<Result<Vec<_>>>()?;
                window_functions::return_type(fun, &data_types)
            }
            Expr::AggregateFunction { fun, args, .. } => {
                let data_types = args
                    .iter()
                    .map(|e| e.get_type(schema))
                    .collect::<Result<Vec<_>>>()?;
                aggregates::return_type(fun, &data_types)
            }
            Expr::AggregateUDF { fun, args, .. } => {
                let data_types = args
                    .iter()
                    .map(|e| e.get_type(schema))
                    .collect::<Result<Vec<_>>>()?;
                Ok((fun.return_type)(&data_types)?.as_ref().clone())
            }
            Expr::Not(_)
            | Expr::IsNull(_)
            | Expr::Between { .. }
            | Expr::InList { .. }
            | Expr::IsNotNull(_) => Ok(DataType::Boolean),
            Expr::BinaryExpr {
                ref left,
                ref right,
                ref op,
            } => binary_operator_data_type(
                &left.get_type(schema)?,
                op,
                &right.get_type(schema)?,
            ),
            Expr::Wildcard => Err(DataFusionError::Internal(
                "Wildcard expressions are not valid in a logical query plan".to_owned(),
            )),
            Expr::GetIndexedField { ref expr, key } => {
                let data_type = expr.get_type(schema)?;

                get_indexed_field(&data_type, key).map(|x| x.data_type().clone())
            }
        }
    }

    /// Returns the nullability of the expression based on [ExprSchema].
    ///
    /// Note: [DFSchema] implements [ExprSchema].
    ///
    /// # Errors
    ///
    /// This function errors when it is not possible to compute its
    /// nullability.  This happens when the expression refers to a
    /// column that does not exist in the schema.
    pub fn nullable<S: ExprSchema>(&self, input_schema: &S) -> Result<bool> {
        match self {
            Expr::Alias(expr, _)
            | Expr::Not(expr)
            | Expr::Negative(expr)
            | Expr::Sort { expr, .. }
            | Expr::Between { expr, .. }
            | Expr::InList { expr, .. } => expr.nullable(input_schema),
            Expr::Column(c) => input_schema.nullable(c),
            Expr::Literal(value) => Ok(value.is_null()),
            Expr::Case {
                when_then_expr,
                else_expr,
                ..
            } => {
                // this expression is nullable if any of the input expressions are nullable
                let then_nullable = when_then_expr
                    .iter()
                    .map(|(_, t)| t.nullable(input_schema))
                    .collect::<Result<Vec<_>>>()?;
                if then_nullable.contains(&true) {
                    Ok(true)
                } else if let Some(e) = else_expr {
                    e.nullable(input_schema)
                } else {
                    Ok(false)
                }
            }
            Expr::Cast { expr, .. } => expr.nullable(input_schema),
            Expr::ScalarVariable(_)
            | Expr::TryCast { .. }
            | Expr::ScalarFunction { .. }
            | Expr::ScalarUDF { .. }
            | Expr::WindowFunction { .. }
            | Expr::AggregateFunction { .. }
            | Expr::AggregateUDF { .. } => Ok(true),
            Expr::IsNull(_) | Expr::IsNotNull(_) => Ok(false),
            Expr::BinaryExpr {
                ref left,
                ref right,
                ..
            } => Ok(left.nullable(input_schema)? || right.nullable(input_schema)?),
            Expr::Wildcard => Err(DataFusionError::Internal(
                "Wildcard expressions are not valid in a logical query plan".to_owned(),
            )),
            Expr::GetIndexedField { ref expr, key } => {
                let data_type = expr.get_type(input_schema)?;
                get_indexed_field(&data_type, key).map(|x| x.is_nullable())
            }
        }
    }

    /// Returns the name of this expression based on [crate::logical_plan::DFSchema].
    ///
    /// This represents how a column with this expression is named when no alias is chosen
    pub fn name(&self, input_schema: &DFSchema) -> Result<String> {
        create_name(self, input_schema)
    }

    /// Returns a [arrow::datatypes::Field] compatible with this expression.
    pub fn to_field(&self, input_schema: &DFSchema) -> Result<DFField> {
        match self {
            Expr::Column(c) => Ok(DFField::new(
                c.relation.as_deref(),
                &c.name,
                self.get_type(input_schema)?,
                self.nullable(input_schema)?,
            )),
            _ => Ok(DFField::new(
                None,
                &self.name(input_schema)?,
                self.get_type(input_schema)?,
                self.nullable(input_schema)?,
            )),
        }
    }

    /// Wraps this expression in a cast to a target [arrow::datatypes::DataType].
    ///
    /// # Errors
    ///
    /// This function errors when it is impossible to cast the
    /// expression to the target [arrow::datatypes::DataType].
    pub fn cast_to<S: ExprSchema>(
        self,
        cast_to_type: &DataType,
        schema: &S,
    ) -> Result<Expr> {
        // TODO(kszucs): most of the operations do not validate the type correctness
        // like all of the binary expressions below. Perhaps Expr should track the
        // type of the expression?
        let this_type = self.get_type(schema)?;
        if this_type == *cast_to_type {
            Ok(self)
        } else if can_cast_types(&this_type, cast_to_type) {
            Ok(Expr::Cast {
                expr: Box::new(self),
                data_type: cast_to_type.clone(),
            })
        } else {
            Err(DataFusionError::Plan(format!(
                "Cannot automatically convert {:?} to {:?}",
                this_type, cast_to_type
            )))
        }
    }

    /// Return `self == other`
    pub fn eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Eq, other)
    }

    /// Return `self != other`
    pub fn not_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::NotEq, other)
    }

    /// Return `self > other`
    pub fn gt(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Gt, other)
    }

    /// Return `self >= other`
    pub fn gt_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::GtEq, other)
    }

    /// Return `self < other`
    pub fn lt(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Lt, other)
    }

    /// Return `self <= other`
    pub fn lt_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::LtEq, other)
    }

    /// Return `self && other`
    pub fn and(self, other: Expr) -> Expr {
        binary_expr(self, Operator::And, other)
    }

    /// Return `self || other`
    pub fn or(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Or, other)
    }

    /// Return `!self`
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Expr {
        !self
    }

    /// Calculate the modulus of two expressions.
    /// Return `self % other`
    pub fn modulus(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Modulo, other)
    }

    /// Return `self LIKE other`
    pub fn like(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Like, other)
    }

    /// Return `self NOT LIKE other`
    pub fn not_like(self, other: Expr) -> Expr {
        binary_expr(self, Operator::NotLike, other)
    }

    /// Return `self AS name` alias expression
    pub fn alias(self, name: &str) -> Expr {
        Expr::Alias(Box::new(self), name.to_owned())
    }

    /// Return `self IN <list>` if `negated` is false, otherwise
    /// return `self NOT IN <list>`.a
    pub fn in_list(self, list: Vec<Expr>, negated: bool) -> Expr {
        Expr::InList {
            expr: Box::new(self),
            list,
            negated,
        }
    }

    /// Return `IsNull(Box(self))
    #[allow(clippy::wrong_self_convention)]
    pub fn is_null(self) -> Expr {
        Expr::IsNull(Box::new(self))
    }

    /// Return `IsNotNull(Box(self))
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_null(self) -> Expr {
        Expr::IsNotNull(Box::new(self))
    }

    /// Create a sort expression from an existing expression.
    ///
    /// ```
    /// # use datafusion::logical_plan::col;
    /// let sort_expr = col("foo").sort(true, true); // SORT ASC NULLS_FIRST
    /// ```
    pub fn sort(self, asc: bool, nulls_first: bool) -> Expr {
        Expr::Sort {
            expr: Box::new(self),
            asc,
            nulls_first,
        }
    }

    /// Performs a depth first walk of an expression and
    /// its children, calling [`ExpressionVisitor::pre_visit`] and
    /// `visitor.post_visit`.
    ///
    /// Implements the [visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern) to
    /// separate expression algorithms from the structure of the
    /// `Expr` tree and make it easier to add new types of expressions
    /// and algorithms that walk the tree.
    ///
    /// For an expression tree such as
    /// ```text
    /// BinaryExpr (GT)
    ///    left: Column("foo")
    ///    right: Column("bar")
    /// ```
    ///
    /// The nodes are visited using the following order
    /// ```text
    /// pre_visit(BinaryExpr(GT))
    /// pre_visit(Column("foo"))
    /// pre_visit(Column("bar"))
    /// post_visit(Column("bar"))
    /// post_visit(Column("bar"))
    /// post_visit(BinaryExpr(GT))
    /// ```
    ///
    /// If an Err result is returned, recursion is stopped immediately
    ///
    /// If `Recursion::Stop` is returned on a call to pre_visit, no
    /// children of that expression are visited, nor is post_visit
    /// called on that expression
    ///
    pub fn accept<V: ExpressionVisitor>(&self, visitor: V) -> Result<V> {
        let visitor = match visitor.pre_visit(self)? {
            Recursion::Continue(visitor) => visitor,
            // If the recursion should stop, do not visit children
            Recursion::Stop(visitor) => return Ok(visitor),
        };

        // recurse (and cover all expression types)
        let visitor = match self {
            Expr::Alias(expr, _)
            | Expr::Not(expr)
            | Expr::IsNotNull(expr)
            | Expr::IsNull(expr)
            | Expr::Negative(expr)
            | Expr::Cast { expr, .. }
            | Expr::TryCast { expr, .. }
            | Expr::Sort { expr, .. }
            | Expr::GetIndexedField { expr, .. } => expr.accept(visitor),
            Expr::Column(_)
            | Expr::ScalarVariable(_)
            | Expr::Literal(_)
            | Expr::Wildcard => Ok(visitor),
            Expr::BinaryExpr { left, right, .. } => {
                let visitor = left.accept(visitor)?;
                right.accept(visitor)
            }
            Expr::Between {
                expr, low, high, ..
            } => {
                let visitor = expr.accept(visitor)?;
                let visitor = low.accept(visitor)?;
                high.accept(visitor)
            }
            Expr::Case {
                expr,
                when_then_expr,
                else_expr,
            } => {
                let visitor = if let Some(expr) = expr.as_ref() {
                    expr.accept(visitor)
                } else {
                    Ok(visitor)
                }?;
                let visitor = when_then_expr.iter().try_fold(
                    visitor,
                    |visitor, (when, then)| {
                        let visitor = when.accept(visitor)?;
                        then.accept(visitor)
                    },
                )?;
                if let Some(else_expr) = else_expr.as_ref() {
                    else_expr.accept(visitor)
                } else {
                    Ok(visitor)
                }
            }
            Expr::ScalarFunction { args, .. }
            | Expr::ScalarUDF { args, .. }
            | Expr::AggregateFunction { args, .. }
            | Expr::AggregateUDF { args, .. } => args
                .iter()
                .try_fold(visitor, |visitor, arg| arg.accept(visitor)),
            Expr::WindowFunction {
                args,
                partition_by,
                order_by,
                ..
            } => {
                let visitor = args
                    .iter()
                    .try_fold(visitor, |visitor, arg| arg.accept(visitor))?;
                let visitor = partition_by
                    .iter()
                    .try_fold(visitor, |visitor, arg| arg.accept(visitor))?;
                let visitor = order_by
                    .iter()
                    .try_fold(visitor, |visitor, arg| arg.accept(visitor))?;
                Ok(visitor)
            }
            Expr::InList { expr, list, .. } => {
                let visitor = expr.accept(visitor)?;
                list.iter()
                    .try_fold(visitor, |visitor, arg| arg.accept(visitor))
            }
        }?;

        visitor.post_visit(self)
    }

    /// Performs a depth first walk of an expression and its children
    /// to rewrite an expression, consuming `self` producing a new
    /// [`Expr`].
    ///
    /// Implements a modified version of the [visitor
    /// pattern](https://en.wikipedia.org/wiki/Visitor_pattern) to
    /// separate algorithms from the structure of the `Expr` tree and
    /// make it easier to write new, efficient expression
    /// transformation algorithms.
    ///
    /// For an expression tree such as
    /// ```text
    /// BinaryExpr (GT)
    ///    left: Column("foo")
    ///    right: Column("bar")
    /// ```
    ///
    /// The nodes are visited using the following order
    /// ```text
    /// pre_visit(BinaryExpr(GT))
    /// pre_visit(Column("foo"))
    /// mutatate(Column("foo"))
    /// pre_visit(Column("bar"))
    /// mutate(Column("bar"))
    /// mutate(BinaryExpr(GT))
    /// ```
    ///
    /// If an Err result is returned, recursion is stopped immediately
    ///
    /// If [`false`] is returned on a call to pre_visit, no
    /// children of that expression are visited, nor is mutate
    /// called on that expression
    ///
    pub fn rewrite<R>(self, rewriter: &mut R) -> Result<Self>
    where
        R: ExprRewriter,
    {
        let need_mutate = match rewriter.pre_visit(&self)? {
            RewriteRecursion::Mutate => return rewriter.mutate(self),
            RewriteRecursion::Stop => return Ok(self),
            RewriteRecursion::Continue => true,
            RewriteRecursion::Skip => false,
        };

        // recurse into all sub expressions(and cover all expression types)
        let expr = match self {
            Expr::Alias(expr, name) => Expr::Alias(rewrite_boxed(expr, rewriter)?, name),
            Expr::Column(_) => self.clone(),
            Expr::ScalarVariable(names) => Expr::ScalarVariable(names),
            Expr::Literal(value) => Expr::Literal(value),
            Expr::BinaryExpr { left, op, right } => Expr::BinaryExpr {
                left: rewrite_boxed(left, rewriter)?,
                op,
                right: rewrite_boxed(right, rewriter)?,
            },
            Expr::Not(expr) => Expr::Not(rewrite_boxed(expr, rewriter)?),
            Expr::IsNotNull(expr) => Expr::IsNotNull(rewrite_boxed(expr, rewriter)?),
            Expr::IsNull(expr) => Expr::IsNull(rewrite_boxed(expr, rewriter)?),
            Expr::Negative(expr) => Expr::Negative(rewrite_boxed(expr, rewriter)?),
            Expr::Between {
                expr,
                low,
                high,
                negated,
            } => Expr::Between {
                expr: rewrite_boxed(expr, rewriter)?,
                low: rewrite_boxed(low, rewriter)?,
                high: rewrite_boxed(high, rewriter)?,
                negated,
            },
            Expr::Case {
                expr,
                when_then_expr,
                else_expr,
            } => {
                let expr = rewrite_option_box(expr, rewriter)?;
                let when_then_expr = when_then_expr
                    .into_iter()
                    .map(|(when, then)| {
                        Ok((
                            rewrite_boxed(when, rewriter)?,
                            rewrite_boxed(then, rewriter)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let else_expr = rewrite_option_box(else_expr, rewriter)?;

                Expr::Case {
                    expr,
                    when_then_expr,
                    else_expr,
                }
            }
            Expr::Cast { expr, data_type } => Expr::Cast {
                expr: rewrite_boxed(expr, rewriter)?,
                data_type,
            },
            Expr::TryCast { expr, data_type } => Expr::TryCast {
                expr: rewrite_boxed(expr, rewriter)?,
                data_type,
            },
            Expr::Sort {
                expr,
                asc,
                nulls_first,
            } => Expr::Sort {
                expr: rewrite_boxed(expr, rewriter)?,
                asc,
                nulls_first,
            },
            Expr::ScalarFunction { args, fun } => Expr::ScalarFunction {
                args: rewrite_vec(args, rewriter)?,
                fun,
            },
            Expr::ScalarUDF { args, fun } => Expr::ScalarUDF {
                args: rewrite_vec(args, rewriter)?,
                fun,
            },
            Expr::WindowFunction {
                args,
                fun,
                partition_by,
                order_by,
                window_frame,
            } => Expr::WindowFunction {
                args: rewrite_vec(args, rewriter)?,
                fun,
                partition_by: rewrite_vec(partition_by, rewriter)?,
                order_by: rewrite_vec(order_by, rewriter)?,
                window_frame,
            },
            Expr::AggregateFunction {
                args,
                fun,
                distinct,
            } => Expr::AggregateFunction {
                args: rewrite_vec(args, rewriter)?,
                fun,
                distinct,
            },
            Expr::AggregateUDF { args, fun } => Expr::AggregateUDF {
                args: rewrite_vec(args, rewriter)?,
                fun,
            },
            Expr::InList {
                expr,
                list,
                negated,
            } => Expr::InList {
                expr: rewrite_boxed(expr, rewriter)?,
                list: rewrite_vec(list, rewriter)?,
                negated,
            },
            Expr::Wildcard => Expr::Wildcard,
            Expr::GetIndexedField { expr, key } => Expr::GetIndexedField {
                expr: rewrite_boxed(expr, rewriter)?,
                key,
            },
        };

        // now rewrite this expression itself
        if need_mutate {
            rewriter.mutate(expr)
        } else {
            Ok(expr)
        }
    }
}

impl Not for Expr {
    type Output = Self;

    fn not(self) -> Self::Output {
        Expr::Not(Box::new(self))
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Expr::BinaryExpr {
                ref left,
                ref right,
                ref op,
            } => write!(f, "{} {} {}", left, op, right),
            Expr::AggregateFunction {
                /// Name of the function
                ref fun,
                /// List of expressions to feed to the functions as arguments
                ref args,
                /// Whether this is a DISTINCT aggregation or not
                ref distinct,
            } => fmt_function(f, &fun.to_string(), *distinct, args, true),
            Expr::ScalarFunction {
                /// Name of the function
                ref fun,
                /// List of expressions to feed to the functions as arguments
                ref args,
            } => fmt_function(f, &fun.to_string(), false, args, true),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Controls how the visitor recursion should proceed.
pub enum Recursion<V: ExpressionVisitor> {
    /// Attempt to visit all the children, recursively, of this expression.
    Continue(V),
    /// Do not visit the children of this expression, though the walk
    /// of parents of this expression will not be affected
    Stop(V),
}

/// Encode the traversal of an expression tree. When passed to
/// `Expr::accept`, `ExpressionVisitor::visit` is invoked
/// recursively on all nodes of an expression tree. See the comments
/// on `Expr::accept` for details on its use
pub trait ExpressionVisitor: Sized {
    /// Invoked before any children of `expr` are visisted.
    fn pre_visit(self, expr: &Expr) -> Result<Recursion<Self>>;

    /// Invoked after all children of `expr` are visited. Default
    /// implementation does nothing.
    fn post_visit(self, _expr: &Expr) -> Result<Self> {
        Ok(self)
    }
}

/// Controls how the [ExprRewriter] recursion should proceed.
pub enum RewriteRecursion {
    /// Continue rewrite / visit this expression.
    Continue,
    /// Call [mutate()] immediately and return.
    Mutate,
    /// Do not rewrite / visit the children of this expression.
    Stop,
    /// Keep recursive but skip mutate on this expression
    Skip,
}

/// Trait for potentially recursively rewriting an [`Expr`] expression
/// tree. When passed to `Expr::rewrite`, `ExpressionVisitor::mutate` is
/// invoked recursively on all nodes of an expression tree. See the
/// comments on `Expr::rewrite` for details on its use
pub trait ExprRewriter: Sized {
    /// Invoked before any children of `expr` are rewritten /
    /// visited. Default implementation returns `Ok(RewriteRecursion::Continue)`
    fn pre_visit(&mut self, _expr: &Expr) -> Result<RewriteRecursion> {
        Ok(RewriteRecursion::Continue)
    }

    /// Invoked after all children of `expr` have been mutated and
    /// returns a potentially modified expr.
    fn mutate(&mut self, expr: Expr) -> Result<Expr>;
}

fn fmt_function(
    f: &mut fmt::Formatter,
    fun: &str,
    distinct: bool,
    args: &[Expr],
    display: bool,
) -> fmt::Result {
    let args: Vec<String> = match display {
        true => args.iter().map(|arg| format!("{}", arg)).collect(),
        false => args.iter().map(|arg| format!("{:?}", arg)).collect(),
    };

    // let args: Vec<String> = args.iter().map(|arg| format!("{:?}", arg)).collect();
    let distinct_str = match distinct {
        true => "DISTINCT ",
        false => "",
    };
    write!(f, "{}({}{})", fun, distinct_str, args.join(", "))
}

#[allow(clippy::boxed_local)]
fn rewrite_boxed<R>(boxed_expr: Box<Expr>, rewriter: &mut R) -> Result<Box<Expr>>
where
    R: ExprRewriter,
{
    // TODO: It might be possible to avoid an allocation (the
    // Box::new) below by reusing the box.
    let expr: Expr = *boxed_expr;
    let rewritten_expr = expr.rewrite(rewriter)?;
    Ok(Box::new(rewritten_expr))
}

fn rewrite_option_box<R>(
    option_box: Option<Box<Expr>>,
    rewriter: &mut R,
) -> Result<Option<Box<Expr>>>
where
    R: ExprRewriter,
{
    option_box
        .map(|expr| rewrite_boxed(expr, rewriter))
        .transpose()
}

/// rewrite a `Vec` of `Expr`s with the rewriter
fn rewrite_vec<R>(v: Vec<Expr>, rewriter: &mut R) -> Result<Vec<Expr>>
where
    R: ExprRewriter,
{
    v.into_iter().map(|expr| expr.rewrite(rewriter)).collect()
}

/// return a new expression l <op> r
pub fn binary_expr(l: Expr, op: Operator, r: Expr) -> Expr {
    Expr::BinaryExpr {
        left: Box::new(l),
        op,
        right: Box::new(r),
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Alias(expr, alias) => write!(f, "{:?} AS {}", expr, alias),
            Expr::Column(c) => write!(f, "{}", c),
            Expr::ScalarVariable(var_names) => write!(f, "{}", var_names.join(".")),
            Expr::Literal(v) => write!(f, "{:?}", v),
            Expr::Case {
                expr,
                when_then_expr,
                else_expr,
                ..
            } => {
                write!(f, "CASE ")?;
                if let Some(e) = expr {
                    write!(f, "{:?} ", e)?;
                }
                for (w, t) in when_then_expr {
                    write!(f, "WHEN {:?} THEN {:?} ", w, t)?;
                }
                if let Some(e) = else_expr {
                    write!(f, "ELSE {:?} ", e)?;
                }
                write!(f, "END")
            }
            Expr::Cast { expr, data_type } => {
                write!(f, "CAST({:?} AS {:?})", expr, data_type)
            }
            Expr::TryCast { expr, data_type } => {
                write!(f, "TRY_CAST({:?} AS {:?})", expr, data_type)
            }
            Expr::Not(expr) => write!(f, "NOT {:?}", expr),
            Expr::Negative(expr) => write!(f, "(- {:?})", expr),
            Expr::IsNull(expr) => write!(f, "{:?} IS NULL", expr),
            Expr::IsNotNull(expr) => write!(f, "{:?} IS NOT NULL", expr),
            Expr::BinaryExpr { left, op, right } => {
                write!(f, "{:?} {} {:?}", left, op, right)
            }
            Expr::Sort {
                expr,
                asc,
                nulls_first,
            } => {
                if *asc {
                    write!(f, "{:?} ASC", expr)?;
                } else {
                    write!(f, "{:?} DESC", expr)?;
                }
                if *nulls_first {
                    write!(f, " NULLS FIRST")
                } else {
                    write!(f, " NULLS LAST")
                }
            }
            Expr::ScalarFunction { fun, args, .. } => {
                fmt_function(f, &fun.to_string(), false, args, false)
            }
            Expr::ScalarUDF { fun, ref args, .. } => {
                fmt_function(f, &fun.name, false, args, false)
            }
            Expr::WindowFunction {
                fun,
                args,
                partition_by,
                order_by,
                window_frame,
            } => {
                fmt_function(f, &fun.to_string(), false, args, false)?;
                if !partition_by.is_empty() {
                    write!(f, " PARTITION BY {:?}", partition_by)?;
                }
                if !order_by.is_empty() {
                    write!(f, " ORDER BY {:?}", order_by)?;
                }
                if let Some(window_frame) = window_frame {
                    write!(
                        f,
                        " {} BETWEEN {} AND {}",
                        window_frame.units,
                        window_frame.start_bound,
                        window_frame.end_bound
                    )?;
                }
                Ok(())
            }
            Expr::AggregateFunction {
                fun,
                distinct,
                ref args,
                ..
            } => fmt_function(f, &fun.to_string(), *distinct, args, true),
            Expr::AggregateUDF { fun, ref args, .. } => {
                fmt_function(f, &fun.name, false, args, false)
            }
            Expr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                if *negated {
                    write!(f, "{:?} NOT BETWEEN {:?} AND {:?}", expr, low, high)
                } else {
                    write!(f, "{:?} BETWEEN {:?} AND {:?}", expr, low, high)
                }
            }
            Expr::InList {
                expr,
                list,
                negated,
            } => {
                if *negated {
                    write!(f, "{:?} NOT IN ({:?})", expr, list)
                } else {
                    write!(f, "{:?} IN ({:?})", expr, list)
                }
            }
            Expr::Wildcard => write!(f, "*"),
            Expr::GetIndexedField { ref expr, key } => {
                write!(f, "({:?})[{}]", expr, key)
            }
        }
    }
}

/// Returns a readable name of an expression based on the input schema.
/// This function recursively transverses the expression for names such as "CAST(a > 2)".
fn create_name(e: &Expr, input_schema: &DFSchema) -> Result<String> {
    match e {
        Expr::Alias(_, name) => Ok(name.clone()),
        Expr::Column(c) => Ok(c.flat_name()),
        Expr::ScalarVariable(variable_names) => Ok(variable_names.join(".")),
        Expr::Literal(value) => Ok(format!("{:?}", value)),
        Expr::BinaryExpr { left, op, right } => {
            let left = create_name(left, input_schema)?;
            let right = create_name(right, input_schema)?;
            Ok(format!("{} {} {}", left, op, right))
        }
        Expr::Case {
            expr,
            when_then_expr,
            else_expr,
        } => {
            let mut name = "CASE ".to_string();
            if let Some(e) = expr {
                let e = create_name(e, input_schema)?;
                name += &format!("{} ", e);
            }
            for (w, t) in when_then_expr {
                let when = create_name(w, input_schema)?;
                let then = create_name(t, input_schema)?;
                name += &format!("WHEN {} THEN {} ", when, then);
            }
            if let Some(e) = else_expr {
                let e = create_name(e, input_schema)?;
                name += &format!("ELSE {} ", e);
            }
            name += "END";
            Ok(name)
        }
        Expr::Cast { expr, data_type } => {
            let expr = create_name(expr, input_schema)?;
            Ok(format!("CAST({} AS {:?})", expr, data_type))
        }
        Expr::TryCast { expr, data_type } => {
            let expr = create_name(expr, input_schema)?;
            Ok(format!("TRY_CAST({} AS {:?})", expr, data_type))
        }
        Expr::Not(expr) => {
            let expr = create_name(expr, input_schema)?;
            Ok(format!("NOT {}", expr))
        }
        Expr::Negative(expr) => {
            let expr = create_name(expr, input_schema)?;
            Ok(format!("(- {})", expr))
        }
        Expr::IsNull(expr) => {
            let expr = create_name(expr, input_schema)?;
            Ok(format!("{} IS NULL", expr))
        }
        Expr::IsNotNull(expr) => {
            let expr = create_name(expr, input_schema)?;
            Ok(format!("{} IS NOT NULL", expr))
        }
        Expr::GetIndexedField { expr, key } => {
            let expr = create_name(expr, input_schema)?;
            Ok(format!("{}[{}]", expr, key))
        }
        Expr::ScalarFunction { fun, args, .. } => {
            create_function_name(&fun.to_string(), false, args, input_schema)
        }
        Expr::ScalarUDF { fun, args, .. } => {
            create_function_name(&fun.name, false, args, input_schema)
        }
        Expr::WindowFunction {
            fun,
            args,
            window_frame,
            partition_by,
            order_by,
        } => {
            let mut parts: Vec<String> = vec![create_function_name(
                &fun.to_string(),
                false,
                args,
                input_schema,
            )?];
            if !partition_by.is_empty() {
                parts.push(format!("PARTITION BY {:?}", partition_by));
            }
            if !order_by.is_empty() {
                parts.push(format!("ORDER BY {:?}", order_by));
            }
            if let Some(window_frame) = window_frame {
                parts.push(format!("{}", window_frame));
            }
            Ok(parts.join(" "))
        }
        Expr::AggregateFunction {
            fun,
            distinct,
            args,
            ..
        } => create_function_name(&fun.to_string(), *distinct, args, input_schema),
        Expr::AggregateUDF { fun, args } => {
            let mut names = Vec::with_capacity(args.len());
            for e in args {
                names.push(create_name(e, input_schema)?);
            }
            Ok(format!("{}({})", fun.name, names.join(",")))
        }
        Expr::InList {
            expr,
            list,
            negated,
        } => {
            let expr = create_name(expr, input_schema)?;
            let list = list.iter().map(|expr| create_name(expr, input_schema));
            if *negated {
                Ok(format!("{} NOT IN ({:?})", expr, list))
            } else {
                Ok(format!("{} IN ({:?})", expr, list))
            }
        }
        Expr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let expr = create_name(expr, input_schema)?;
            let low = create_name(low, input_schema)?;
            let high = create_name(high, input_schema)?;
            if *negated {
                Ok(format!("{} NOT BETWEEN {} AND {}", expr, low, high))
            } else {
                Ok(format!("{} BETWEEN {} AND {}", expr, low, high))
            }
        }
        Expr::Sort { .. } => Err(DataFusionError::Internal(
            "Create name does not support sort expression".to_string(),
        )),
        Expr::Wildcard => Err(DataFusionError::Internal(
            "Create name does not support wildcard".to_string(),
        )),
    }
}

fn create_function_name(
    fun: &str,
    distinct: bool,
    args: &[Expr],
    input_schema: &DFSchema,
) -> Result<String> {
    let names: Vec<String> = args
        .iter()
        .map(|e| create_name(e, input_schema))
        .collect::<Result<_>>()?;
    let distinct_str = match distinct {
        true => "DISTINCT ",
        false => "",
    };
    Ok(format!("{}({}{})", fun, distinct_str, names.join(",")))
}
