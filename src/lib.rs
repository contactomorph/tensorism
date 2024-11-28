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
//! Macros `⟨xxx⟩_make` expect an expression in which character `$` has a special meaning.
//! It is to follow a sequence of identifiers separated by spaces. Such identifiers are
//! called **[Ricci](https://en.wikipedia.org/wiki/Gregorio_Ricci-Curbastro) indexes**, like
//! `i` and `j` in the following example:
//! ```
//! let x = ndarray_make! {i j $ a[i, j] + b[j]};
//! ```
//! Ricci indexes alway represent variables of type `usize`. They are successively assigned values from 0
//! to the maximum possible integer. This latter value is determine by the arrays for which they are used
//! as indexes. In the case of `i` in the previous example, the maximum assigned value is determine by the
//! first dimension of array `a`. In the case of `j`, the second dimension of `a` and the only dimension of
//! `b` must agree. If it is not the code `panic!` at runtime.
//! 
//! # Generating new arrays
//! 
//! When Ricci indexes are used at the very beginning of a `⟨xxx⟩_make` macro, the result is 
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
