= Tensorism =

A small experimental library for manipulating arrays with multiple indexes. It is meant to be:
 * Concise: Specific macros can be used to easily express transformations in a form similar to 
   the related mathematical expressions.
 * Type-safe: Compatibility of dimensions can be checked at compilation time.

== Overview ==

Tensorism is divided into two sibling crates:
 * [tensorism](https://docs.rs/tensorism-gen/latest/tensorism/) contains types and traits.
 * [tensorism-gen](https://docs.rs/tensorism-gen/latest/tensorism-gen/) contains macros to
   efficiently write formulas.

== Examples ==

```rust
let matrix4x4: Tensor2<StaticDimTag<4>, StaticDimTag<4>, f64> = …;
let trace: tensorism_gen::make!(std::iter::Sum.sum(i # matrix4x4[i, i]));
```
$$\mathtt{trace} \leftarrow \sum_i \\mathtt{matrix4x4}_{i, i}$$

```rust
let matrixA: Tensor2<StaticDimTag<4>, StaticDimTag<4>, f64> = …;
let matrixB: Tensor2<StaticDimTag<4>, StaticDimTag<4>, f64> = …;
let matricC: tensorism_gen::make!(i k # std::iter::Sum.sum(i j k # matrixA[i, j] * matrixB[j, k]));
```
$$\forall i, \forall k, \mathtt{matrixC}_{i, k} \leftarrow \mathtt{matrixA}_{i, j} \cdot \mathtt{matrixB}_{j, k}$$