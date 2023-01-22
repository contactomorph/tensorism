# Tensorism

A small experimental library for manipulating arrays with multiple indexes. It is meant to be:
 * Concise: Specific macros can be used to easily express transformations in a form similar to 
   the related mathematical expressions.
 * Type-safe: Compatibility of dimensions can be checked at compilation time.

## Overview

Tensorism is divided into two sibling crates:
 * **tensorism** (this library) contains types and traits.
 * [tensorism-gen](https://crates.io/crates/tensorism-gen) contains macros to
   efficiently write formulas.

## Examples

* Computing the trace of a matrix:
```rust
use tensorism_gen::make;
use std::iter::Sum;
let mM: Tensor2<StaticDimTag<4>, StaticDimTag<4>, f64> = …;

let tau = make!(<f64>::sum(i $ mM[i, i])); // Or equivalently: `make!((i $ mM[i, i]).sum())`
```
$$\tau \leftarrow \sum_{i=0}^3 M_{i, i}$$

* Multiplying two matrices:
```rust
use tensorism_gen::make;
use std::iter::Sum;
let mA: Tensor2<StaticDimTag<7>, StaticDimTag<4>, Complex64> = …;
let mB: Tensor2<StaticDimTag<4>, StaticDimTag<5>, Complex64> = …;

let mC = make!(i k $ <Complex64>::sum(j $ mA[i, j] * mB[j, k]));
```
$$\forall i \in 0 .. 7,\quad \forall k \in 0 .. 5,\quad C_{i, k} \leftarrow \sum_{0 \leq j < 4} A_{i, j} \cdot B_{j, k}$$

* Finding the maximum values (here almong instants) according to given "axes":
```rust
use tensorism_gen::make;
use datetime::Instant;
fn maximum_of(it: impl Iterator<Item=Instant>) -> Instant { … }
let tD: Tensor3<StaticDimTag<10>, StaticDimTag<25>, StaticDimTag<3>, Instant> = …;

let mX = make!(i j $ maximum_of(k $ tD[i, j, k])); // : Tensor2<StaticDimTag<10>, StaticDimTag<25>, Instant>
let mY = make!(k i $ maximum_of(j $ tD[i, j, k])); // : Tensor2<StaticDimTag<3>, StaticDimTag<10>, Instant>
let v = make!(j $ maximum_of(i k $ tD[i, j, k])); // : Tensor1<StaticDimTag<25>, Instant>
let d = make!(maximum_of(i j k $ tD[i, j, k])); // : Instant
```
$$\forall i \in 0 .. 10,\quad \forall j \in 0 .. 25,\quad X_{i, j} \leftarrow \max_{0 \leq k < 3} D_{i, j, k}$$

$$\forall k \in 0 .. 3,\quad \forall i \in 0 .. 10,\quad Y_{k, i} \leftarrow \max_{0 \leq j < 25} D_{i, j, k}$$

$$\forall j \in 0 .. 25,\quad v_j \leftarrow \underset{0 \leq k < 3}{\max_{0 \leq i < 10}} D_{i, j, k}$$

$$d \leftarrow \underset{0 \leq k < 3}{\underset{0 \leq j < 25}{\max_{0 \leq i < 10}}} D_{i, j, k}$$

* Computing intersections:
```rust
use tensorism_gen::make;
use std::string::String;
use std::collections::HashSet;
fn intersection_of<'a>(it: impl Iterator<Item=&'a HashSet<String>>) -> HashSet<String> { … }
let mA: Tensor2<StaticDimTag<4>, StaticDimTag<3>, HashSet<String>> = …;

let u = make!(i $ intersection_of(j $ &mA[i, j])); // : Tensor1<StaticDimTag<4>, HashSet<String>>
let v = make!(j $ intersection_of(i $ &mA[i, j])); // : Tensor1<StaticDimTag<3>, HashSet<String>>
```
$$\forall i \in 0 .. 4,\quad u_j \leftarrow \bigcap_{0 \leq j < 3} A_{i, j}$$

$$\forall j \in 0 .. 3,\quad v_j \leftarrow \bigcap_{0 \leq i < 4} A_{i, j}$$

* Computing logical conjunctions and disjunctions:
```rust
use tensorism_gen::make;
fn forall(it: impl Iterator<Item=bool>) -> bool { … }
fn exists(it: impl Iterator<Item=bool>) -> bool { … }
let q: Tensor3<StaticDimTag<3>, StaticDimTag<9>, StaticDimTag<7>, bool> = …;

let p = make!(k $ forall(i $ exists(j $ q[i, j, k])));
```
$$\forall k \in 0 .. 7,\quad p_k \leftarrow \Big( \forall i \in 0 .. 3, \ \exists j \in 0 .. 9, \ q_{i, j, k} \Big)$$

* Any combination:
```rust
use tensorism_gen::make;
let q: Tensor3<StaticDimTag<3>, StaticDimTag<9>, StaticDimTag<7>, bool> = …;
let mA: Tensor2<StaticDimTag<3>, StaticDimTag<13>, f64> = …;
let mB: Tensor2<StaticDimTag<7>, StaticDimTag<13>, f64> = …;
let mLambda: Tensor2<StaticDimTag<7>, StaticDimTag<13>, f64> = …;
fn product(it: impl Iterator<Item=f64>) -> f64 { … }

let mZ = make!(i k $ if exists(j $ q[i, j, k] && 0f64 <= mA[i, j]) {
    <f64>::sum(l $ mLambda[k, l] * sin(2 * PI * mA[i, l]))
  } else {
    product(m $ mA[i, m] + mB[k, m])
  });
```
$$\forall i \in 0 .. 3,\quad \forall k \in 0 .. 7,\quad Z_{i, k} \leftarrow \begin{cases}
  \sum_{0 \leq l < 13} \Lambda_{k, l} \cdot \sin\left(2 \cdot \pi \cdot A_{i, l} \right) & \mathrm{if} & \exists j \in 0 .. 9, \ q_{i, j, k} \wedge 0 \leq A_{i, j}\\
  \prod_{0 \leq m < 13} (A_{i, m} + B_{k, m}) & \mathrm{else}
\end{cases}$$
