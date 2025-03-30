extern crate proc_macro;
#[macro_use]
extern crate quote;

use proc_macro2::{Literal, TokenStream, TokenTree};

mod parsing;
mod sequentialization;
mod types;

use parsing::parse;
use quote::ToTokens;
use sequentialization::sequentialize;

fn simplify(text: &str) -> String {
    let mut result = String::new();
    text.split('\n').map(|s| s.trim()).for_each(|s| {
        result.push_str(s);
        result.push(' ')
    });
    result
}

#[proc_macro]
pub fn ndarray_make(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match parse(input) {
        Err(invalid_stream) => invalid_stream.into(),
        Ok((sequence, index_use, tensor_use)) => {
            sequentialize(sequence, index_use, tensor_use).into()
        }
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
            TokenTree::Literal(Literal::string(string.as_str())).to_tokens(&mut output);
            output.into()
        }
    }
}
