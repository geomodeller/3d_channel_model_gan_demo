# Generative Adversarial Networks Demonstration on 3D Channelized Reservoir Models

### Written by Honggeun Jo, Assistant Professor, Inha University (Korea)

This demonstration showcases the implementation of Generative Adversarial Networks (GANs) to extract geological features from a 3D channelized reservoir model. The model dimensions are $\(60 \times 60 \times 7\) along the \(x\)-, \(y\)-, and \(z\)-directions.$ The example 3D channel model, commonly referred to as the "Egg Model," was originally introduced by [Jansen et al. (2014)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/gdj3.21). 

### Implemented GAN Models

1. **Unconditional GAN (Vanilla GAN):**  
   - The generator and discriminator are implemented using simple convolutional neural networks with batch normalization and dropout layers.  
   - The trained generator produces permeability models based on latent variables consisting of 100 numeric values.

2. **Conditional GAN (cGAN):**  
   - The generator is built using a U-Net architecture, designed to take an input permeability model and generate the corresponding facies model.

### Additional Features

- **Reservoir Simulation Integration:**  
  Simulation data from SLB's Eclipse is included for those who wish to link the generated permeability models to reservoir simulations.

- **Hands-on Exploration:**  
  You can experiment with the models using the provided data. Note that the training dataset is not included due to its size. If needed, please contact me for access.

### Contact Information

Feel free to reach out with any questions or requests:  
- **Email:** honggeun.jo@inha.ac.kr  
- **YouTube:** [whghdrms](https://www.youtube.com/@whghdrms)  
- **GitHub:** [whghdrms](https://github.com/whghdrms)  
- **Google Scholar:** [Profile](https://scholar.google.com/citations?user=u0OE5CIAAAAJ&hl=en)  
- **LinkedIn:** [Profile](https://www.linkedin.com/in/honggeun-jo/)  
