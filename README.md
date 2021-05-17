# vq
A fast and lightweight vector quantization library/toolkit entirely written in and for Rust

Currently implements:
- Vector Quantization (VQ, [1]) (```vq::VectorQuantization```)
- Learning Vector Quantization (LVQ, [2]) (```vq::LearningVectorQuantization```)
- General Learning Vector Quantization (GLVQ, [3])(```vq::GeneralLearningVectorQuantization```)
- General Matrix Learning Vector Quantization (GMLVQ, [4]) (```vq::GeneralMatrixLearningVectorQuantization```)

Each model supports access to the learned parameters, the ability to reproduce runs through a seeded RNG and the ability to set a custom learning rate scheduler.

## Documentation
To access the documentation simply run:
```cargo doc --open```

## References

[1] Dana H. Ballard (2000). An Introduction to Natural Computation. MIT Press. p. 189.

[2] T. Kohonen (1995), "Learning vector quantization", in M.A. Arbib (ed.), The Handbook of Brain Theory and Neural Networks, Cambridge, MA: MIT Press, pp. 537–540

[3] Atsushi Sato and Keiji Yamada. 1995. Generalized learning vector quantization. In Proceedings of the 8th International Conference on Neural Information Processing Systems (NIPS'95). MIT Press, Cambridge, MA, USA, 423–429.

[4] P. Schneider, B. Hammer, and M. Biehl (2009). "Adaptive Relevance Matrices in Learning Vector Quantization". Neural Computation. 21 (10): 3532–3561.
