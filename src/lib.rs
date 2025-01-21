//! Macros for handling arrays with multiple indexes.
//!
//! Provides macros for arrays with multiple indexes as provided by popular crates that are:
//! - convenient to manipulate using a custom dsl;
//! - minimising the number of runtime checks about validity of indexes;
//! - available for:
//!   - [ndarray](https://crates.io/crates/ndarray).
//!
//! # The `$` magical character and Ricci indexes
//! 
//! Macros `⟨lib⟩_make` expect an expression in which character `$` has a special meaning.
//! It is to follow a sequence of identifiers separated by spaces. Such identifiers are
//! called **[Ricci](https://en.wikipedia.org/wiki/Gregorio_Ricci-Curbastro) indexes**, like
//! `i` and `j` in the following example:
//! ```
//! let x = ndarray_make! {i j $ a[i, j] + b[j]};
//! ```
//! Ricci indexes alway represent variables of type `usize`. They are successively assigned values from 0
//! to the maximum possible integer. This latter value is determine by the arrays for which they are used
//! as indexes. In the case of `i` in the previous example, the maximum assigned value is determine by the
//! first dimension of array `a` (and is equal to `a.dim().0 - 1`). In the case of `j`, the second dimension
//! of `a` and the only dimension of `b` must agree. If they don't, the code `panic!` at runtime. If they do,
//! the maximum value for `j` is `a.dim().1 - 1`, which is then equal to `b.dim() - 1`.
//! 
//! Note that in the expression after `$`, Ricci indexes are always used to access items using syntax
//! `⟨array⟩[⟨index_1⟩,…,⟨index_d⟩]`, whatever is the actual syntax in the underlying library.
//! 
//! # Generating new arrays
//! 
//! When Ricci indexes are used at the very beginning of the parameter of a `⟨lib⟩_make` macro, the result is
//! a multi-index array with as many indexes. In our previous example `x` has type `ndarray::Array2<T>` where
//! `T` is the type of the expression `a[i, j] + b[j]`.
//! 
//! In the following example, the result `y` is an array with 3 indexes (of type `ndarray::Array3<T>`) generated
//! by evaluating a more complex expression involving arrays `p`, `q`, `r` and `s`.
//! ```
//! let y = ndarray_make! {i j k $ if p[i, j] - 0.3 < 0.4 * q[j, k] { r[j]* q[j, k] + 0.2 } else { 0.5 * s[i, j, k] }};
//! ```
//! 
//! # Aggregating arrays values
//! 
//! When Ricci indexes are used in a sub-expression of the parameter of a `⟨lib⟩_make` macro, this sub-expression
//! evaluate as an iterator. Iterator values
#![feature(proc_macro_quote)]
#![feature(extend_one)]

extern crate proc_macro;
#[macro_use]
extern crate quote;

use proc_macro2::{Literal, TokenStream, TokenTree};

mod parsing;
mod sequentialization;
mod types;

use parsing::parse;
use sequentialization::sequentialize;

fn simplify(text: &String) -> String {
    let mut result = String::new();
    text.split('\n').map(|s| s.trim()).for_each(|s| {
        result.extend_one(s);
        result.extend_one(' ')
    });
    result
}

/// Macro that generate a new ndarray::Array by computing its expression.
#[proc_macro]
pub fn ndarray_make(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match parse(input) {
        Err(invalid_stream) => invalid_stream.into(),
        Ok((sequence, index_use, tensor_use)) => sequentialize(sequence, index_use, tensor_use).into(),
    }
}

#[doc(hidden)]
#[proc_macro]
pub fn ndarray_format_for_make(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match parse(input) {
        Err(invalid_stream) => invalid_stream.into(),
        Ok((sequence, index_use, tensor_use)) => {
            let output = sequentialize(sequence, index_use, tensor_use);
            let string = simplify(&output.to_string());
            let mut output = TokenStream::new();
            output.extend_one(TokenTree::Literal(Literal::string(string.as_str())));
            output.into()
        }
    }
}
