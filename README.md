# vq
A fast and lightweight vector quantization library/toolkit entirely written in and for Rust

Currently implements:
- Vector Quantization (```vq::VectorQuantization```)
- Learning Vector Quantization (```vq::LearningVectorQuantization```)
- General Learning Vector Quantization (```vq::GeneralLearningVectorQuantization```)
- General Matrix Learning Vector Quantization (```vq::GeneralMatrixLearningVectorQuantization```)

Each model supports access to the learned parameters, the ability to reproduce runs through a seeded RNG and the ability to set a custom learning rate scheduler.

## Documentation
To access the documentation simply run:
```cargo doc --open```
