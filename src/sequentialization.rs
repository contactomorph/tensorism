use crate::types::*;
use proc_macro2::{Delimiter, Group, Literal, Span, TokenStream, TokenTree};
use quote::ToTokens;

fn sequentialize_tensor_func(func: RicciFunction, stream: &mut TokenStream) {
    let mut direct_indexes = func.inverted_indexes.clone();
    direct_indexes.reverse();
    let indexes_tuple = quote! {(#(#direct_indexes),*, )};
    let mut mappings = indexes_tuple.clone();
    for (i, index) in func.inverted_indexes.into_iter().enumerate() {
        let dimension_name = format_ident!("{}_dimension", index);
        let span = index.span();
        mappings = if i == 0 {
            quote_spanned! {span => (0usize..#dimension_name).map(move |#index| #mappings)}
        } else {
            quote_spanned! {span => (0usize..#dimension_name).flat_map(move |#index| #mappings)}
        }
    }
    let span = func.sequence.span;
    let mut content_stream = TokenStream::new();
    sequentialize_sequence(func.sequence, &mut content_stream);
    let func_stream = quote_spanned! {
        span =>  #mappings.map(|#indexes_tuple| { #content_stream })
    };
    stream.extend(func_stream);
}

fn sequentialize_sequence(sequence: RicciSequence, stream: &mut TokenStream) {
    let mut content = TokenStream::new();
    for alt in sequence.content.into_iter() {
        match alt {
            RicciAlternative::Func(sub_func) => {
                sequentialize_tensor_func(sub_func, &mut content);
            }
            RicciAlternative::Tree(token) => token.to_tokens(&mut content),
            RicciAlternative::Seq(sub_sequence) => {
                sequentialize_sequence(sub_sequence, &mut content);
            }
            RicciAlternative::TensorAccess {
                tensor_name,
                span,
                indexes,
            } => {
                let stream = if indexes.len() == 1 {
                    let index = &indexes[0];
                    quote_spanned! {
                        span => (* unsafe{ #tensor_name.uget(#index) })
                    }
                } else {
                    quote_spanned! {
                        span => (* unsafe{ #tensor_name.uget((#(#indexes, )*)) })
                    }
                };
                content.extend(stream);
            }
        }
    }
    if sequence.use_parens {
        TokenTree::Group(Group::new(Delimiter::Parenthesis, content)).to_tokens(stream);
    } else {
        stream.extend(content)
    }
}

fn create_dimension_computation_token_stream(
    tensor_use: &TensorUse,
    ricci_position: &RicciPosition,
) -> TokenStream {
    let tensor_name = &ricci_position.tensor_name;
    if tensor_use.get_order(tensor_name) == 1 {
        quote_spanned! {
            tensor_name.span() => ::ndarray::ArrayBase::<_, _>::dim(&#tensor_name)
        }
    } else {
        let pos = Literal::usize_unsuffixed(ricci_position.position);
        quote_spanned! {
            tensor_name.span() => ::ndarray::ArrayBase::<_, _>::dim(&#tensor_name).#pos
        }
    }
}

fn sequentialize_header(index_use: IndexUse, tensor_use: &TensorUse) -> TokenStream {
    let mut output = TokenStream::new();
    for (name, positions) in index_use.into_iter() {
        let dimension_name = format_ident!("{}_dimension", name);
        let ricci_position = positions.first().unwrap();
        let dimension_computation =
            create_dimension_computation_token_stream(tensor_use, ricci_position);
        let length_definition = quote_spanned! {
            ricci_position.tensor_name.span() => let #dimension_name: usize = #dimension_computation;
        };
        output.extend(length_definition);
        for ricci_position in positions.into_iter().skip(1) {
            let other_dimension_computation =
                create_dimension_computation_token_stream(tensor_use, &ricci_position);
            let equality_assertion = quote_spanned! {
                ricci_position.index_name.span() =>
                {
                    let first: usize = #dimension_computation;
                    let second: usize = #other_dimension_computation;
                    if first != second { panic!("Non matching dimensions") }
                }
            };
            output.extend(equality_assertion);
        }
    }
    output
}

fn try_extract_func(mut sequence: RicciSequence) -> Result<RicciFunction, RicciSequence> {
    if let [RicciAlternative::Func(_)] = sequence.content.as_slice() {
        if let Some(RicciAlternative::Func(func)) = sequence.content.pop() {
            if func.inverted_indexes.is_empty() {
                Err(func.sequence)
            } else {
                Ok(func)
            }
        } else {
            panic!("Unreachable")
        }
    } else {
        Err(sequence)
    }
}

fn sequentialize_shape_creation(mut func: RicciFunction) -> TokenStream {
    let mut direct_indexes = func.inverted_indexes.drain(..).collect::<Vec<_>>();
    direct_indexes.reverse();
    let dimensions = direct_indexes
        .iter()
        .map(|i| format_ident!("{}_dimension", i))
        .collect::<Vec<_>>();
    let mut substream = TokenStream::new();
    let order = dimensions.len();
    sequentialize_sequence(func.sequence, &mut substream);
    quote_spanned! {
        Span::call_site() =>
            ::ndarray::Array::<_, ::ndarray::Dim<[::ndarray::Ix; #order]>>::from_shape_fn(
                (#(#dimensions),*, ),
                |(#(#direct_indexes),*, )| { #substream }
            )
    }
}

fn sequentialize_body(sequence: RicciSequence, stream: &mut TokenStream) {
    match try_extract_func(sequence) {
        Ok(func) => stream.extend(sequentialize_shape_creation(func)),
        Err(sequence) => sequentialize_sequence(sequence, stream),
    }
}

pub fn sequentialize(
    sequence: RicciSequence,
    index_use: IndexUse,
    tensor_use: TensorUse,
) -> TokenStream {
    let mut stream = sequentialize_header(index_use, &tensor_use);
    sequentialize_body(sequence, &mut stream);
    let mut output = TokenStream::new();
    TokenTree::Group(Group::new(Delimiter::Brace, stream)).to_tokens(&mut output);
    output
}
