use crate::types::*;
use proc_macro2::{Delimiter, Group, Ident, Punct, Span, TokenStream, TokenTree};

fn generate_compilation_error(span: Span, message: &'static str) -> Result<(), TokenStream> {
    let invalid_stream = quote_spanned! {
        span => compile_error!(#message)
    };
    Err(invalid_stream)
}

fn parse_punct(
    punct: Punct,
    tokens: &mut impl Iterator<Item = TokenTree>,
    sequence: &mut RicciSequence,
    index_use: &mut IndexUse,
    tensor_use: &mut TensorUse,
) -> Result<(), TokenStream> {
    let c = punct.as_char();
    if c == ';' {
        generate_compilation_error(punct.span(), "Character ';' is forbidden")?
    } else if c == '$' {
        let inverted_indexes = sequence.extract_previous_identifiers();
        let mut direct_indexes = inverted_indexes.clone();
        direct_indexes.reverse();
        for index_name in direct_indexes {
            let added = index_use.declare_new(index_name.clone());
            if !added {
                generate_compilation_error(index_name.span(), "Illegal reused index name")?
            }
        }
        let mut new_sequence = RicciSequence::naked(punct.span());
        parse_sequence(tokens, &mut new_sequence, index_use, tensor_use)?;
        let func = RicciFunction::new(inverted_indexes, new_sequence);
        sequence.content.push(RicciAlternative::Func(func));
    } else {
        sequence.push_token(TokenTree::Punct(punct.clone()));
    }
    Ok(())
}

fn parse_tensor_indexing(group: Group) -> Result<Vec<Ident>, TokenStream> {
    let mut indexes = Vec::new();
    for token in group.stream() {
        match token {
            TokenTree::Ident(index) => indexes.push(index),
            TokenTree::Punct(p) => {
                if p.as_char() != ',' {
                    generate_compilation_error(group.span(), "Invalid content in indexes")?
                }
            }
            _ => generate_compilation_error(group.span(), "Invalid content in indexes")?,
        }
    }
    Ok(indexes)
}

fn parse_group(
    group: Group,
    sequence: &mut RicciSequence,
    index_use: &mut IndexUse,
    tensor_use: &mut TensorUse,
) -> Result<(), TokenStream> {
    match group.delimiter() {
        Delimiter::Brace => {
            generate_compilation_error(group.span(), "Characters '{' and '}' are forbidden")?
        }
        Delimiter::Bracket => {
            if let Some(RicciAlternative::Tree(TokenTree::Ident(tensor_name))) =
                sequence.content.last()
            {
                let tensor_name = tensor_name.clone();
                let span = group.span();
                let indexes = parse_tensor_indexing(group)?;
                for (position, index_name) in indexes.iter().enumerate() {
                    let added = index_use.push(index_name.clone(), tensor_name.clone(), position);
                    if !added {
                        generate_compilation_error(index_name.span(), "Undeclared index")?
                    }
                }
                let consistent = tensor_use.update_order(&tensor_name, indexes.len());
                if !consistent {
                    generate_compilation_error(tensor_name.span(), "Inconsistent number of indexes.")?
                }
                sequence.content.pop();
                sequence.content.push(RicciAlternative::TensorAccess {
                    tensor_name,
                    span,
                    indexes,
                });
            } else {
                generate_compilation_error(
                    group.span(),
                    "Invalid tensor name: an identifier was expected",
                )?
            }
        }
        delimiter => {
            let mut new_sequence = if delimiter == Delimiter::Parenthesis {
                RicciSequence::with_parens(group.span())
            } else {
                RicciSequence::naked(group.span())
            };
            parse_sequence(
                &mut group.stream().into_iter(),
                &mut new_sequence,
                index_use,
                tensor_use,
            )?;
            sequence.content.push(RicciAlternative::Seq(new_sequence));
        }
    }
    Ok(())
}

fn parse_sequence(
    tokens: &mut impl Iterator<Item = TokenTree>,
    sequence: &mut RicciSequence,
    index_use: &mut IndexUse,
    tensor_use: &mut TensorUse,
) -> Result<(), TokenStream> {
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Punct(punct) => parse_punct(punct, tokens, sequence, index_use, tensor_use)?,
            TokenTree::Group(group) => parse_group(group, sequence, index_use, tensor_use)?,
            _ => sequence.push_token(token),
        }
    }
    Ok(())
}

pub fn parse(input: proc_macro::TokenStream) -> Result<(RicciSequence, IndexUse, TensorUse), TokenStream> {
    let input: TokenStream = input.into();
    let mut sequence = RicciSequence::initial();
    let mut index_use = IndexUse::new();
    let mut tensor_use = TensorUse::new();
    parse_sequence(&mut input.into_iter(), &mut sequence, &mut index_use, &mut tensor_use)?;
    Ok((sequence, index_use, tensor_use))
}
