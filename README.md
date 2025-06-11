# [Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation](https://ryanxli.github.io/reflect3d/index.html) 

CVPR, 2025 (Highlight ✨)

[[Project Page]](https://ryanxli.github.io/reflect3d/index.html)  [[Paper]](http://arxiv.org/abs/2411.17763)

https://github.com/user-attachments/assets/5f4febbc-b26a-429b-b6c7-0a9fcd9d8bd1

**TL;DR**: We propose Reflect3D, a zero-shot single-image 3D reflection symmetry detector; we improve single-image 3D generation through symmetry-aware optimization.

## Code Release Plan

- [ ] 🎯 By the end of Jun 2025: Symmetry detection dataset, code, and checkpoints.
      
    - [ ] 📦 Dataset
          
    - [ ] 🤖 Inference
          
    - [ ] 🏋️‍♂️ Training
         
- [ ] 🎯 By the end of July 2025: Symmetry conditioned 3D generation.

## Abstract

Symmetry is a ubiquitous and fundamental property in the visual world, serving as a critical cue for perception and structure interpretation. This paper investigates the detection of 3D reflection symmetry from a single RGB image, and reveals its significant benefit on single-image 3D generation.

We introduce Reflect3D, a scalable, zero-shot symmetry detector capable of robust generalization to diverse and real-world scenarios. Inspired by the success of foundation models, our method scales up symmetry detection with a transformer-based architecture. We also leverage generative priors from multi-view diffusion models to address the inherent ambiguity in single-view symmetry detection. Extensive evaluations on various data sources demonstrate that Reflect3D establishes a new state-of-the-art in single-image symmetry detection.

Furthermore, we show the practical benefit of incorporating detected symmetry into single-image 3D generation pipelines through a symmetry-aware optimization process. The integration of symmetry significantly enhances the structural accuracy, cohesiveness, and visual fidelity of the reconstructed 3D geometry and textures, advancing the capabilities of 3D content creation.

