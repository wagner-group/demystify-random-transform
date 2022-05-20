# Demystifying the Adversarial Robustness of Random Transformation Defenses

AAAI 2022 AdvML Workshop (Best Paper, Oral Presentation) [link](https://openreview.net/forum?id=p4SrFydwO5)  
ICML 2022 (Short Talk)
<!-- [ArXiv]() -->

## Abstract

Neural networks' lack of robustness against attacks raises concerns in security-sensitive settings such as autonomous vehicles. While many countermeasures may look promising, only a few withstand rigorous evaluation. Defenses using random transformations (RT) have shown impressive results, particularly BaRT (Raff et al., 2019) on ImageNet. However, this type of defense has not been rigorously evaluated, leaving its robustness properties poorly understood. Their stochastic properties make evaluation more challenging and render many proposed attacks on deterministic models inapplicable. First, we show that the BPDA attack (Athalye et al., 2018a) used in BaRT’s evaluation is ineffective and likely over-estimates its robustness. We then attempt to construct the strongest possible RT defense through the informed selection of transformations and Bayesian optimization for tuning their parameters. Furthermore, we create the strongest possible attack to evaluate our RT defense. Our new attack vastly outperforms the baseline, reducing the accuracy by 83% compared to the 19% reduction by the commonly used EoT attack (4.3x improvement). Our result indicates that the RT defense on Imagenette dataset (ten-class subset of ImageNet) is not robust against adversarial examples. Extending the study further, we use our new attack to adversarially train RT defense (called AdvRT), resulting in a large robustness gain.

## Requirement

See `environment.yml`.

- pytorch >= 1.6
- torchvision >= 0.6.0
- numpy >= 1.18.1
- pyyaml >= 5.3.1
- kornia >= 0.4.2 (commit: [61af3bcd83b19cb64b9a62b1fc0a0aeb17094b31](https://github.com/kornia/kornia/tree/61af3bcd83b19cb64b9a62b1fc0a0aeb17094b31))

```bash
# Install dependencies
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pyyaml scikit-image pip
pip install -U ray ray[tune] kornia torch_optimizer bayesian-optimization

# Load Imagenette dataset
mkdir data && cd data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvf imagenette2-320.tgz
```

## File Organization

- There are multiple scripts to run the training and the testing. Main portion of code is in `./adv`.
- The naming of the scripts is simply `<SCRIPT>_<DATASET>.py` with the YAML config file under the same name. `<DATASET>` includes `cifar` (CIFAR-10), and `imagenette` (10-class subset ImageNet).
- Main scripts
  - `train-test.py`: main script for training and testing. The options and hyperparameters can be set in `train-test_DATASET.yml`.
  - `test.py`: test a trained network under an attack. Set parameters in `test_<DATASET>.yml`.
  - `train_ray.py`: run Bayesian optimization experiments. Set parameters in `train_ray_<DATASET>.yml`.
- Important files in the library `./adv/`
  - `attacks/`: implements the attacks
    - `pgd_attack.py`: implements PGD attack.
    - `opt_attack.py`: implements optimization-based attack.
    - `rand_pgd_attack.py`: implements optimization-based attack for RT models.
    - `rand_opt_attack.py`: implements PGD attack for RT models.
  - `utils/`: general utility files
    - `dataset_utils.py`: handles dataset loading.
    - `diversity.py`: computes ensemble diversity metrics.
    - `train_utils.py`: helper functions for training.
    - `test_utils.py`: helper functions for training.
    - `utils.py`: miscellaneous utility files.
  - `wrappers/`: wrapper model on top of the networks
    - `transforms/`: implement all transformations
      - `jpeg/`: implement differentiable JPEG compression
    - `rand_wrapper.py`: implement random transform wrapper for image datasets.
    - `wrapper_utils.py`: helper function for applying wrapper models.

## Usage

- We use YAML config files (`.yml`) for all training and testing scripts. The path to this config file must be specified as an argument when running the Python script.
- See `test_example.yml` for parameter descriptions used during testing.
- Weights and hyperparameters of the transformations of the best RT model can be found via this [link](https://drive.google.com/file/d/1UYV3LYyv34h5UvNkhxH_QQvzqWiSfToA/view?usp=sharing).

### Examples

To test an RT model trained on Imagenette,

```bash
python test.py configs/test_img_rand.yml
```
