# Adversarial attacks and defences on CIFAR-10

*This is an English translation of the report submitted with the original
coursework. It is kept as a record of what was written at the time, so its
numbers are the ones from the original run and its descriptions are the
original descriptions. Where the report and the code it describes disagree, or
where a later reading of the code changed the conclusion, a note marks the
spot — the surrounding text is left as it was. The
[README](../README.md#what-was-corrected) has the full list.*

---

## Part 1 — Attacks

*Figure 1: several test images the baseline model already classifies
incorrectly.* — [`assets/baseline-misclassified.png`](../assets/baseline-misclassified.png)

At the end of this part the methods are compared with one another and their
strengths and weaknesses stated.

### FGSM

The aim of the Fast Gradient Sign Method is to apply a small but deliberately
chosen change to the model's input, fooling it into an incorrect decision.
Using the gradient of the cost function with respect to the input, the method
finds the direction that produces the largest increase in error and moves the
image along it. The formula is

$$x_{adv} = x + \epsilon \cdot \operatorname{sign}(\nabla_x J(\theta, x, y))$$

where $x$ is the original image, $x_{adv}$ the manipulated (adversarial)
image, $\epsilon$ a small value setting the strength of the attack, and
$J(\theta, x, y)$ the model's cost function with parameters $\theta$. Taking
the sign of the gradient guarantees that the change is applied only in the
direction that leads to the greatest error in the model's output.

To produce an adversarial image the gradient of the cost with respect to the
model's input must be computed. It is obtained by backpropagation and written

$$\nabla_x J(\theta, x, y)$$

where

- $J(\theta, x, y)$ — the model's cost function (cross-entropy, for example)
- $\theta$ — the model's parameters, its weights
- $x$ — the input image
- $y$ — the correct label for the input

In practice the model first produces its output for input $x$; calling
`loss.backward()` then computes the gradient of the cost with respect to $x$
and stores it. That gradient is used in the next step to build the adversarial
image.

### FGSM results

*Figure 2: results of the FGSM attack.* —
[`assets/attack-fgsm-examples.png`](../assets/attack-fgsm-examples.png)

- **Accuracy on normal images: 92.59%.** The model performs well, as expected,
  on the kind of images it was trained on.
- **Accuracy on adversarial images (FGSM, $\epsilon = 0.03$): 15.18%.** After
  the attack the model's accuracy has dropped sharply, showing its
  vulnerability to small but targeted changes in the input.
- **Accuracy drop: 77.41%.** This severe fall is evidence of the attack's
  effectiveness and of the model's inability to recognise manipulated images.

In the figure, the first row holds the original images, all correctly
classified by the model. The second row holds the perturbation maps generated
from the gradient of the cost function; although these perturbations are
nearly invisible to the human eye, they are enough to mislead the model. The
third row shows the final images after the perturbation has been added, where
the model's recognition fails — an image labelled `1`, for instance, is
predicted as `9`.

> **Note.** This FGSM applies $\epsilon = 0.03$ in **pixel** space, which is
> `7.65/255`. The PGD, momentum-PGD and AutoPGD cells that follow apply the
> same number in **normalised** space, which is about `1.5/255` in pixels —
> roughly a fifth of the budget. The comparison table in this report presents
> all five attacks as though they shared one $\epsilon$; they do not. The
> qualitative conclusion survives, and in fact gets stronger: PGD reaches 7.08%
> while FGSM only reaches 15.18% *despite* having a five times smaller budget.

### PGD (Projected Gradient Descent)

This method is a generalised, multi-step version of FGSM. The central idea is
to change the input image gradually through several successive updates in the
direction that increases the model's error, while keeping those changes inside
a fixed region — a ball of radius $\epsilon$ in $\ell_\infty$ — so they never
exceed the permitted bound.

The update is

$$x^{t+1} = \Pi_{B_\epsilon(x)}\big(x^t + \alpha \cdot
\operatorname{sign}(\nabla_x J(\theta, x^t, y))\big)$$

where

- $x^0$ — the initial image, which may start with random noise
- $x^{t+1}$ — the adversarial image at step $t + 1$
- $\alpha$ — the update step size at each stage, smaller than $\epsilon$
- $J(\theta, x, y)$ — the model's cost function
- $\Pi_{B_\epsilon(x)}$ — projection of the image onto the nearest point
  inside the $\ell_\infty$ ball of radius $\epsilon$; a clipping operation

Unlike FGSM, which runs in a single step, PGD repeats this process over
several stages, changing the input image gradually in the direction of
increasing error. After each stage a projection or clamp operation returns the
image to the permitted region, preventing it from drifting too far.

### PGD results

*Figure 3: results of the PGD attack.* —
[`assets/attack-pgd-examples.png`](../assets/attack-pgd-examples.png)

- **Accuracy on normal images: 92.59%**
- **Accuracy on adversarial images (PGD, $\epsilon = 0.03$): 7.08%.** Faced
  with PGD the model's accuracy collapses and it misclassifies almost every
  sample.
- **Accuracy drop: 85.51%.** This large fall in performance shows that PGD is
  far more powerful than FGSM, because it uses a multi-step, iterative attack.

In the figure, the first column shows the original image without perturbation.
The second column shows the perturbation map, added to the input gradually
over several stages. The third column is the final adversarial image, which to
the human eye looks almost unchanged and which the model nevertheless
misclassifies.

### PGD with momentum

One important improvement to PGD is the use of momentum in the gradient update.
The technique makes the trajectory through input space smoother and prevents
unwanted oscillation.

In ordinary PGD each step uses only the gradient at that instant to update the
input. In the momentum version the current gradient is combined with the
gradients of previous steps so that the path is more stable and more effective.
The normalised gradient with momentum is computed as

$$g^{t+1} = \mu \cdot g^t +
\frac{\nabla_x J(\theta, x^t, y)}{\lVert \nabla_x J(\theta, x^t, y) \rVert_1}$$

and the input is then updated using the combined gradient:

$$x^{t+1} = \Pi_{B_\epsilon(x)}\big(x^t + \alpha \cdot
\operatorname{sign}(g^{t+1})\big)$$

where

- $g^t$ — the momentum vector at step $t$
- $\mu$ — the momentum coefficient
- $\alpha$ — the step size at each stage
- $\Pi$ — projection onto the permitted region, such as the $\ell_\infty$ ball

Advantages of momentum:

- greater stability of the update path and less gradient oscillation
- a stronger attack than plain PGD even at the same number of steps
- a lower chance of getting stuck in flat regions of the optimisation space

> **Note.** The formula above is the MI-FGSM update of Dong et al. (2018) and is
> correct as written: the gradient is divided by its own L1 norm before being
> accumulated. The code that produced the numbers below accumulates
> `grad.sign()` instead. Since `sign` discards magnitude, the normalisation that
> makes $\mu$ meaningful across steps is missing, and the momentum term does
> little beyond decay. The corrected implementation in
> [`advlab.mifgsm`](../advlab.py) follows the formula.

### PGD with momentum — results

*Figure 4: results of the PGD + momentum attack.* —
[`assets/attack-pgd-momentum-examples.png`](../assets/attack-pgd-momentum-examples.png)

- **Accuracy on normal images: 92.59%**
- **Accuracy on adversarial images (PGD + momentum, $\epsilon = 0.03$):
  4.72%.** The attack causes a striking fall in accuracy and shows the model to
  be severely affected, having practically lost the ability to recognise
  correctly.
- **Accuracy drop: 87.87%**

### AutoPGD with the CE and CW loss functions

AutoPGD (Auto Projected Gradient Descent) is an advanced, adaptive version of
PGD which, unlike PGD, requires no manual choice of step size. The algorithm
runs the attack automatically in two main phases:

- **Exploration** — the attack begins with relatively large steps so that the
  search is guided towards regions with effective, useful gradients.
- **Exploitation** — the step size is then adjusted dynamically according to
  the attack's progress, to help maximise the error.

How AutoPGD proceeds:

1. The attack starts from the original image.
2. A step is taken in the direction of increasing error, using the gradient of
   the cost function.
3. The step size is adaptively adjusted or reduced according to how much the
   objective has improved.
4. Momentum is used to stabilise the path.
5. At fixed checkpoints the algorithm reviews its progress. If no improvement
   has been achieved in a given period, it continues from the best previous
   point.

The loss functions used:

- **Cross-entropy (CE)** — suited to standard attacks based on producing an
  incorrect prediction.
- **Carlini & Wagner (CW)** — suited to more targeted attacks, focused on the
  model's decision boundary.

Advantages of AutoPGD:

- adaptive execution with no need to tune parameters by hand
- an effective combination of exploration and exploitation phases, raising the
  chance of a successful attack
- momentum and periodic review, for greater stability and precision

> **Note.** The description above is an accurate account of Croce & Hein (2020).
> The code that produced the numbers below implements none of it: there is no
> adaptive step size, no momentum, no checkpoint schedule and no restart from
> the best point — it is plain fixed-step PGD run for 20 iterations at
> $\alpha = 0.01$, which is why its numbers land close to the PGD row. A real
> AutoPGD is implemented in [`advlab.apgd`](../advlab.py).

### AutoPGD with CE and CW — results

*Figure 5: results of the AutoPGD attack — CE above, CW below.* —
[`assets/attack-apgd-ce-examples.png`](../assets/attack-apgd-ce-examples.png),
[`assets/attack-apgd-cw-examples.png`](../assets/attack-apgd-cw-examples.png)

- **Accuracy on normal images: 92.59%**
- **Accuracy after AutoPGD with cross-entropy (CE): 2.54%.** This severe fall
  shows the model to be highly vulnerable to this kind of attack.
- **Accuracy after AutoPGD with Carlini & Wagner (CW): 2.98%.** Although a
  little higher than the previous case, the drop in accuracy is still very
  large and shows the model to be easily fooled.

Both loss functions used in the AutoPGD algorithm were able to disable the
model effectively. The CW-based attack was slightly less destructive, but the
difference is negligible in practice and both methods are highly efficient at
fooling the model.

The table below shows the model's accuracy after each kind of adversarial
attack. All the attacks were run on one ResNet-20 model trained on CIFAR-10,
with $\epsilon = 0.03$.

| attack | accuracy after attack (%) | notes |
|---|---|---|
| FGSM | 15.18 | fast, single-step; simple but relatively weak |
| PGD | 7.08 | iterative and stronger than FGSM |
| PGD + momentum | 4.72 | improved with momentum for gradient stability; high penetration |
| AutoPGD (CE) | 2.54 | adaptive attack with the cross-entropy cost; very effective |
| AutoPGD (CW) | 2.98 | more targeted and precise, with the Carlini & Wagner function |
| none (normal images) | 92.59 | the model's baseline performance, with no disturbance |

*Table 1: comparison of the model's accuracy after different adversarial
attacks.*

Judging by the table, AutoPGD — with either the CE or the CW function —
produced the largest drop in accuracy and is the strongest attack in this
comparison. The use of momentum also strengthened the effect of PGD. FGSM, by
contrast, having only a single step, is the simplest and least effective of the
methods examined.

> **Note.** As recorded above, the five rows of this table were not measured at
> a common $\epsilon$. The four PGD-family rows share one budget; the FGSM row
> was measured at roughly five times that. The ordering AutoPGD > momentum-PGD
> > PGD > FGSM is nevertheless the ordering these attacks have at a common
> budget — the mismatch only understates how much weaker FGSM is.

### A closer comparison of strengths and weaknesses

The previous table showed the difference in the model's accuracy against the
various attacks. The advantages and disadvantages of each method are examined
more precisely below.

**FGSM (Fast Gradient Sign Method)**

- *Advantages:* very fast to run — a single gradient computation is enough.
  Suitable for a first test of a model's resistance.
- *Disadvantages:* low destructive power; often ineffective against models
  that have been made even slightly robust.

**PGD (Projected Gradient Descent)**

- *Advantages:* multi-step; far stronger than FGSM and effective at reducing
  the model's accuracy.
- *Disadvantages:* slower than FGSM because of the iteration; requires manual
  tuning of parameters such as the step size.
- *Appropriate use:* comprehensive and precise evaluation of a model's
  resistance.

**PGD with momentum (momentum-PGD)**

- *Advantages:* greater stability in the direction of the gradient; a higher
  chance of escaping flat points.
- *Disadvantages:* requires careful tuning of the momentum coefficient $\mu$;
  a higher computational burden.
- *Appropriate use:* targeted attacks with high penetration.

**AutoPGD (adaptive PGD)**

- *Advantages:* adaptive; requires no manual parameter setting; combines the
  exploration and exploitation phases; equipped with a checkpointing
  mechanism.
- *Disadvantages:* more complex to implement and slower to run than the
  simpler attacks.
- *Appropriate use:* final, precise evaluation of how vulnerable a model is.

**The CW loss function in AutoPGD**

- *Advantages:* focuses specifically on the model's decision boundary; is
  effective even against models with some level of defence.
- *Disadvantages:* needs more careful tuning of parameters such as the
  threshold; its computation is somewhat more complex than CE's.
- *Appropriate use:* highly precise, targeted attacks for measuring the
  sensitivity of a model's decision boundaries.

---

## Part 2 — Defences

At the end of this part the methods are compared with one another and their
strengths and weaknesses stated.

### Training with augmented data

In this approach, instead of training on the original data alone, altered and
synthetic versions of the same data are also used. These versions may involve
noise, rotation, colour change, random cropping or translation, produced by
methods such as `rotate`, `color jitter` and `random crop`.

Steps:

1. Fifty per cent of the training images are selected and new data produced by
   applying various transformations.
2. The initial model is fine-tuned again on the combination of original and
   altered data.
3. Finally the model's performance on the test set is measured and recorded.

To gauge the effectiveness of the augmentation, the model is put up against
adversarial attacks such as FGSM and PGD. Its accuracy under normal and
adversarial conditions is then compared with the version from before the
augmentation was applied.

**Feature analysis.** To inspect visually how well clean and adversarial data
are separated, the features extracted from the model — the output before the
fully-connected layer, for example — are displayed in two dimensions using
UMAP. If the two kinds of data are well separated from one another, it is a
sign that the model's understanding of the difference in the inputs' behaviour
has improved.

> **Note.** The description above says the initial model is *fine-tuned* on
> half the training images. The code that produced the numbers below
> constructs the network with `pretrained=False` and trains from scratch for 10
> epochs on the full 50,000-image training set. The explanation offered in the
> next section — that the model "lost its original accuracy because it was
> fine-tuned with augmented data" — is therefore wrong about its own cause:
> there was no original accuracy to lose. 79.00% is simply what a ResNet-20
> reaches from random initialisation in 10 epochs.

### Augmented-data results and the UMAP plot

*Figure 6: analysis of the penultimate features of the model trained with data
augmentation on CIFAR-10. Circular points are clean samples, triangles FGSM,
squares PGD. The colour of each point indicates the image's class.* —
[`assets/umap-augmentation.png`](../assets/umap-augmentation.png),
[`assets/loss-augmentation.png`](../assets/loss-augmentation.png)

Model accuracy:

- **Clean accuracy: 79.00%**
- **FGSM accuracy: 32.11%**
- **PGD accuracy: 18.67%**

The model is seen to have become a little more robust, but it is still weak;
and because it was fine-tuned with augmented data it has lost its original
accuracy.

**UMAP analysis.** The classes are seen to have mixed with one another, and on
the whole no particular grouping can be made out. Only in a very limited way
can scattered groupings be seen for a few specific classes, which indicates
the model's weakness against adversarial attacks.

### Adversarial training

The central idea of this method is very simple: the model is fine-tuned on
images built by the adversarial-image methods.

*Figure 7: UMAP of the penultimate features after adversarial training.
Circular, triangular and square points are clean, FGSM and PGD data
respectively, and the colour of each point represents the corresponding
class.* —
[`assets/umap-adversarial-training.png`](../assets/umap-adversarial-training.png)

Model accuracy:

- **Accuracy on clean data: 86.82%**
- **Accuracy after the FGSM attack: 66.17%**
- **Accuracy after the PGD attack: 62.67%**

**UMAP analysis.** After adversarial training the model has managed to arrange
its embedding so that:

1. *Convergent clustering* — the points belonging to clean and adversarial
   data now fall almost on top of one another, and the class clusters have
   stayed unified and compact. This convergence is a sign that the internal
   structure of the embedding has been preserved against adversarial
   perturbation.
2. *Reduced separation* — unlike the case without adversarial training, where
   the adversarial clusters separated from the clean ones, the FGSM and PGD
   points now lie in the immediate neighbourhood of the clean samples.

*Figure 8: the adversarial-training loss curve.* —
[`assets/loss-adversarial-training.png`](../assets/loss-adversarial-training.png)

3. *Class distinction preserved* — alongside the overlap of clean and
   adversarial, the coloured clusters belonging to the different classes have
   remained separate and distinguishable from one another, which confirms the
   model's classifying power in the embedding space.
4. *The loss curve* — the falling loss shows that the model reduced its
   adversarial error steadily at each epoch and reached a stable embedding.

These results say that adversarial training has both allowed the model to stay
robust in its accuracy on clean data (above 85%) and brought the adversarial
samples closer to the original clusters. The model can therefore be said to
have become properly robust, and its tolerance of the FGSM and PGD attacks has
risen considerably.

### Circle Loss and contrastive learning

**1. Circle Loss.** Circle Loss is an objective function in metric learning
which weights positive and negative points together with dynamic coefficients.
Its central idea is

$$L = \log\Big[1 + \sum_i \exp\big(\gamma\alpha^+_i(s^+_i - \Delta)\big)
+ \sum_j \exp\big(\gamma\alpha^-_j(\Delta - s^-_j)\big)\Big]$$

where

- $s^+_i$ and $s^-_j$ are the similarities — scalars, for instance — of the
  positive and negative samples
- $\alpha^+_i$ and $\alpha^-_j$ are dynamic coefficients controlling the
  closeness of the positives and the distance of the negatives
- $\gamma$ is the scale, the rate of concentration, and $\Delta$ the
  separation threshold

This dynamic weighting means that the more distant positive samples are
reinforced more, and the closer negative samples repelled more sharply.

**2. Contrastive learning.** Contrastive learning is a method which, by pairing
similar samples — two different examples of one class, or two augmentations of
one image — against dissimilar ones, tries to

- bring the positive samples close together in the embedding space, and
- push the negative samples apart.

**Proxy-based Circle Loss**

- A "representative" vector, or proxy, is learned for each class.
- The input $x$ passes through the network and an embedding $e$ is produced.
- The cosine similarity $\ell_c = \gamma\langle e, p_c \rangle$ is computed
  between $e$ and each proxy.
- Circle Loss then adjusts these similarities so that
  - samples come closer to the correct proxy, and
  - move away from the incorrect proxies.

Why we used it:

- *Focus on the hard cases* — "hard" samples, those close to a mistake, take a
  larger weight and are learned faster.
- *Single-stage* — end-to-end; the class prediction is direct.
- *Simultaneous improvement of embedding and class* — the embeddings are
  optimised for more precise separation and higher accuracy in prediction.
- Without this method our model had no defence at all against the attacks.

**The proxy classifier**

- A proxy vector $p_c \in \mathbb{R}^D$ is learned for each class $c$.
- The input $x$ passes through the network and an embedding
  $e \in \mathbb{R}^D$ is produced.
- The similarity between $e$ and each proxy is computed by inner product or
  cosine:

$$s_c = \langle e, p_c \rangle \quad\text{or}\quad
s_c = \frac{e^\top p_c}{\lVert e \rVert \lVert p_c \rVert}$$

- These $s_c$ values serve as logits in a loss function — cross-entropy or
  Circle Loss — for predicting the class.

Advantages:

- *Fast* — only one matrix product $eP^\top$ is needed.
- *Single-stage* — the proxies are updated along with the network in one
  backpropagation.
- *Integrated with metric learning* — with proxy Circle Loss the embedding can
  be moved directly towards or away from the class centres.

### Results (proxy-based Circle Loss)

*Figure 9: UMAP of the features of the layer before `fc`, after training with
Circle Loss.* —
[`assets/umap-proxy-circle-loss.png`](../assets/umap-proxy-circle-loss.png)

*Figure 10: circle loss.* —
[`assets/loss-circle-loss.png`](../assets/loss-circle-loss.png)

Accuracies:

- **Clean data: 90.25%**
- **FGSM attacks: 47.45%**
- **PGD attacks: 41.38%**

**UMAP analysis.**

- The clean points, the green circles, have settled into relatively compact
  and separate clusters, which shows that the proxies have managed to separate
  the clean embeddings well.
- The FGSM points, the orange triangles, have moved some distance from the
  original clusters and penetrated the neighbouring ones; this is what brought
  about the roughly one-half reduction in the model's accuracy against FGSM.
- The PGD points, the grey squares, show the greatest scatter and overlap with
  the other classes, which expresses the model's weakness against multi-step
  PGD attacks.

On the whole the method has managed an acceptable performance, but circle loss
by itself is not enough to make the model robust. It merely keeps the data of
one class close together in feature space, and making the model robust requires
other methods.

> **Note.** Two things about this section.
>
> First, the accuracies given here are labelled as the *proxy* variant's, and
> they match the proxy numbers in the code. The plain pair-based Circle Loss
> experiment also ran and produced 90.05% clean, 33.57% FGSM and 6.64% PGD;
> those numbers appear nowhere in this report. The gap between them is the
> interesting part — the proxy variant is over six times more robust to PGD.
>
> Second, all four numbers in this section, and all four in the autoencoder
> section below, were measured with `torchattacks`' `FGSM` and `PGD` applied to
> **normalised** tensors. Those classes assume their input is an image in
> `[0, 1]` and clamp to that range internally, so every negative value in a
> normalised image — roughly half of them, since normalisation subtracts the
> channel mean — is crushed to zero before the attack begins. The attack these
> numbers measure is not the attack they are labelled as, and they cannot be
> compared with the numbers in Part 1.

### Defending the model with an autoencoder

*Figure 11: UMAP of the embeddings of the model's last layer after the images
have passed through the autoencoder. Circular points are clean, triangles
FGSM, squares PGD; the colours indicate the classes.* —
[`assets/umap-autoencoder-adversarial.png`](../assets/umap-autoencoder-adversarial.png)

*Figure 12: the MSE curve over the autoencoder's 15 training epochs.* —
[`assets/loss-autoencoder-adversarial-mse.png`](../assets/loss-autoencoder-adversarial-mse.png)

Model accuracy:

- **Clean: 44.61%**
- **FGSM: 42.53%**
- **PGD: 25.26%**

**Analysis.**

1. *A small reduction in the distance between clean and adversarial* — as can
   be seen in Figure 11, unlike the adversarially trained model, which made
   the clusters completely overlap, here the FGSM and PGD clusters are still a
   long way from the clean cluster.
2. *The autoencoder's limited effect* — although the MSE fell to some extent
   during training (Figure 12), that reduction in error had no appreciable
   influence on the quality of the embedding or on the separation of the
   adversarial data.
3. *A large drop in accuracy* — the model's accuracy after correction by the
   autoencoder is very low compared with the baseline: only 44.6% for clean
   images and under 43% for FGSM.

The autoencoder, in the role of a simple pre-processor, was unable to create
acceptable resistance to the attacks, because its main objective is a
reduction in reconstruction MSE, not an improvement of the margin between
classes in the embedding space.

Overall the model's accuracy came down, but relative to its original accuracy
it showed good resistance. In this robustification the fully-connected layer
was fine-tuned for several epochs on images cleaned by the autoencoder. To
train the autoencoder we first pass the images produced by the attack
algorithms through it, and to compute the loss they are compared with the clean
data.

> **Note.** Three things about this section.
>
> First, the accuracies here — 44.61 / 42.53 / 25.26 — do not match the ones in
> the notebook that was submitted, which are 45.29 / 50.83 / 25.58. They were
> written from a different run. The difference matters: the submitted
> notebook's FGSM accuracy is *higher than its clean accuracy*, which a working
> defence cannot produce and which this report's own numbers do not show.
>
> Second, the autoencoder's decoder ends in a `Sigmoid`, so its output can only
> lie in `[0, 1]`, while the MSE target is a normalised image whose values run
> from about −2.4 to 2.7. Roughly half of every target is negative by
> construction and unreachable. The MSE plateau at about 0.925 visible in
> Figure 12 is the decoder's output range, not the limit of its capacity.
>
> Third, a second autoencoder was trained — on clean images plus
> `N(0, 0.1)` Gaussian noise rather than on adversarial images — and reached
> 52.85% clean, 48.00% FGSM and 21.29% PGD. It is not mentioned anywhere in
> this report. Its results are close enough to the adversarial autoencoder's to
> suggest that what limits both is the reconstruction objective rather than the
> choice of training corruption.

---

## Summary

- Data augmentation
- Adversarial training
- Contrastive learning with Circle Loss
- Autoencoder-based defence

Each method has its own advantages and disadvantages, and the more appropriate
choice depends on the limits of the available resources, the kind of threat,
and how important it is to preserve accuracy on clean data.

### 1. Data augmentation

*Strengths*

- By applying transformations such as rotation, cropping and colour change,
  the model becomes more resistant to simple perturbations.
- Its accuracy both in the normal case and under attack may improve if the
  method is combined with model weight averaging.

*Weaknesses*

- By itself it is not enough to counter gradient-based attacks such as FGSM and
  PGD; it helps more with reducing overfitting.
- Excessive variety in the augmented data may reduce the model's accuracy or
  stability.

### 2. Adversarial training

*Strengths*

- The most effective method against common attacks such as FGSM and PGD.
- By bringing adversarial samples into the training process, the model finds a
  stronger decision margin.

*Weaknesses*

- Very costly computationally, because of the need to generate adversarial
  data continuously.
- It may suffer robust overfitting: good performance against specific attacks
  but failure against new ones.

### 3. Contrastive learning with Circle Loss

*Strengths*

- Strengthens the distance between classes and compresses the samples of one
  class, with dynamic weighting.
- Simple to implement on the embedding, and able to be combined with
  adversarial training.

*Weaknesses*

- Without the use of adversarial data, only the internal structure of the
  classes improves; overall resistance does not increase.
- Usually needs to be accompanied by cross-entropy or other regularisers.

### 4. Autoencoder-based defence

*Strengths*

- Removes noise by reconstructing the image through compression.
- Lighter to compute than adversarial training.

*Weaknesses*

- The use of MSE or similar criteria provides no guarantee that the decision
  boundary between classes is preserved.
- Performs poorly against strong attacks such as PGD, and the model's accuracy
  may fall dramatically.
