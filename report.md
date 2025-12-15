[2511.10647v1.pdf](https://leedong25.yuque.com/attachments/yuque/0/2025/pdf/45861457/1764639587692-0842c96d-9e20-40ea-aa30-21c47760d036.pdf)_[arXiv preprint arXiv:2511.10647]_

**Github**：[https://github.com/ByteDance-Seed/depth-anything-3?tab=readme-ov-file](https://github.com/ByteDance-Seed/depth-anything-3?tab=readme-ov-file)  
**DeepWiki**：[https://deepwiki.com/ByteDance-Seed/Depth-Anything-3](https://deepwiki.com/ByteDance-Seed/Depth-Anything-3)  
**项目主页及在线 Demo**：[https://depth-anything-3.github.io/](https://depth-anything-3.github.io/)

# 论文内容概述
## 研究背景与动机
<u>从视觉输入中感知和理解 3D 空间信息的能力是人类空间智能的基石，也是机器人和混合现实等应用的关键要求</u>。<u>这种基本需求激发了广泛的 3D 视觉任务，包括单目深度估计 </u>_<u>[23]</u>_<u>，从运动中恢复结构</u>_<u> [80]</u>_<u>，多视角立体视觉 </u>_<u>[73]</u>_<u> 以及同时定位与建图 [58]</u>。<font style="color:#DF2A3F;">（研究背景）</font>尽管这些任务在概念上存在很强的重叠性——通常仅因输入视图数量等单一因素而有所不同——但<u>主流范式一直是为每个任务单独开发高度专业化的模型</u>。虽然<u>最近的努力 </u>_<u>[91, 97]</u>_<u> 已经探索了统一模型以同时处理多个任务，但它们通常存在关键限制：它们往往依赖复杂的、 定制化的架构，通过从头开始对任务进行联合优化进行训练，并因此无法有效利用大规模预训练模型</u>。<font style="color:#DF2A3F;">（研究动机）</font>

> _[23] _**Depth Map Prediction from a Single Image（NIPS 2014）**：<u>提出首个多尺度卷积网络用于单目深度估计</u>，通过粗到细的预测结构显著提升单张 RGB 图像的深度恢复精度，奠定了深度学习在几何感知中的基础性范式。
>
> _[80] _**Photo Tourism: Exploring Photo Collections in 3D（ACM TOG 2006）**：<u>构建利用海量网络照片进行三维重建的完整 SfM 系统</u>，实现大规模非结构化照片集的三维重建，并推动后续大规模视觉 3D 建模方向的发展。
>
> _[73] _**A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms（CVPR 2006）**：<u>建立多视角立体重建（MVS）的系统性基准</u>，通过统一数据、评测协议和算法对比，为后续 MVS 研究提供标准评测框架和技术比较基线。
>
> _[58] _**ORB-SLAM: A Versatile and Accurate Monocular SLAM System（IEEE T-RO 2015）**：<u>提出基于 ORB 特征的实时、鲁棒单目 SLAM 系统</u>，融合跟踪、建图、回环和图优化，在精度与稳定性上成为长期主流基线。
>
> _[91] _**VGGT: Visual Geometry Grounded Transformer（CVPR 2025）**：<u>提出几何约束驱动的 Transformer 框架</u>，将视觉几何先验显式融入模型中，实现统一处理多种 3D 视觉任务，并能有效利用大规模预训练表示。
>
> _[97]_ **DUSt3R: Geometric 3D Vision Made Easy（CVPR 2024）**：<u>提出可从任意两张图像直接预测稠密 pointmaps 和像素匹配的统一模型</u>，具备强泛化性、跨任务能力和鲁棒性，可作为多类 3D 几何任务的基础组件。
>

## Related Work
### Multi-view visual geometry estimation（多视角视觉几何估计）
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765069164428-cd6696ed-3040-4a78-adb3-b8aabe6e9280.png)

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765069219954-8541a36e-2f61-4029-be77-199a81e08562.png)

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765069325011-d99fb10b-b1a0-4bf8-8bb1-22ab6adbae68.png)

以上为网图，非论文图像

<u>传统系统</u> _[70, 71]_ 将重建分解为 特征检测与匹配、鲁棒相对位姿估计、利用 BA 优化的增量式或全局 SfM，以及用于每张图像深度和融合点云的稠密多视角立体。这些方法<u>在纹理良好的场景中仍表现出色，但其</u>模块化和脆弱的对应关系<u>在低纹理、镜面反射或大视角变化的情况下会降低鲁棒性</u>。

> [70] **Structure-from-Motion Revisited（CVPR 2016）**：<u>提出现代 SfM 的统一框架</u>，通过全局/局部 BA、匹配管理与鲁棒估计的系统性整合，显著提升 SfM 的准确性、稳健性与可扩展性，成为经典 SfM 实现（如 COLMAP）的核心基础。
>
> [71] **Pixelwise View Selection for Unstructured Multi-View Stereo（ECCV 2016）**：<u>提出逐像素的视图选择策略，使 MVS 能够在非规则视图布置和大规模数据上更稳定地重建稠密深度</u>，在低纹理与遮挡区域显著提升 PatchMatch 系列方法的鲁棒性。
>

<u>早期的学习方法在组件层面注入了鲁棒性</u>：学习检测器 _[20]_，用于匹配的描述符 _[22]_， 和可微分优化层，这些层将位姿/深度更新暴露给梯度流 _[31, 33, 62]_。在稠密建图方面，用于 MVS 的代价体网络 _[106, 114]_ 取代了手工设计的正则化方法，使用 3D 卷积神经网络，从而在大基线和细结构上相比_经典 PatchMatch_ 提高了深度精度。早期的端到端方法 _[86, 90]_ 通过直接从图像对中回归相机位姿和每张图像的深度，超越了模块化的 SfM/MVS 流程。<u>这些方法降低了工程复杂性，并展示了学习联合深度姿态估计的可行性，但它们往往在可扩展性、泛化能力以及处理任意输入数量方面存在困难</u>。  

> _经典 PatchMatch_： 指传统手工 PatchMatch Stereo 重建方法与其变体。
>

> _[20]_ **SuperPoint: Self-Supervised Interest Point Detection and Description（CVPRW 2018）**：<u>提出自监督的关键点检测与描述符生成网络</u>，通过 Homographic Adaptation 自动生成训练数据，使模型对视角变化更鲁棒，并成为后续特征匹配与 SLAM 的重要基础。
>
> _[22]_ **D2-Net: A Trainable CNN for Joint Detection and Description of Local Features（CVPR 2019）**：<u>提出可训练的端到端特征检测与描述框架，使用同一 CNN 提供检测响应和描述符</u>，在大视角变化情况下显著优于传统手工特征。
>
> _[31]_ **Multi-View Reconstruction via SfM-Guided Monocular Depth Estimation（CVPR 2025）**：<u>提出利用 SfM 指导单目深度学习</u>，从粗 SfM 几何中蒸馏跨视图一致性，使深度预测更符合真实结构，并提升多视图重建质量。
>
> _[33]_ **Detector-Free Structure from Motion（CVPR 2024）**：<u>提出无需显式特征点检测的 SfM 系统，直接从图像对中学习稠密对应关系以进行姿态估计</u>，减少传统 SfM 中特征提取与匹配瓶颈，提升非纹理场景的鲁棒性。
>
> _[62]_ **Global Structure-from-Motion Revisited（ECCV 2024）**：<u>提出改进的全局 SfM 管线</u>，使用更稳健的相对姿态优化与全局旋转/平移同步策略，在数据噪声较大和回环丰富的场景中显著提升重建稳定性。
>
> _[106]_ **Iterative Geometry Encoding Volume for Stereo Matching（CVPR 2023）**：<u>提出迭代式几何编码体（GEV）网络，使用 3D 卷积逐步细化视差估计</u>，在大基线场景与复杂几何结构中超越传统代价体方法。
>
> _[114]_ **MVSNet: Depth Inference for Unstructured Multi-View Stereo（ECCV 2018）**：<u>提出端到端 MVS 网络</u>，使用可微代价体构建与 3D CNN 正则化，实现对无结构多视角输入的高质量深度估计，成为深度学习 MVS 的重要里程碑。
>
> _[86]_ **DeepV2D: Video to Depth with Differentiable Structure-from-Motion（arXiv 2018）**：<u>提出可微分的 SfM 模块，将深度预测与相机姿态更新共同放入端到端优化框架</u>，实现视频序列的联合深度与位姿估计。
>
> _[90]_ **VGG-SfM: Visual Geometry Grounded Deep Structure from Motion（CVPR 2024）**：<u>提出以视觉几何约束驱动的端到端 SfM 模型，通过 Transformer 与几何一致性约束联合训练，实现从图像对中直接回归深度与相对姿态</u>，并提升泛化能力。
>

<u>一个转折点出现在 DUst3R，它利用 Transformer 直接预测两个视图之间的点图，并以纯粹的前馈方式同时计算深度和相对姿态</u>。<u>这项工作为后续基于 Transformer 的方法奠定了基础，</u>这些方法旨在大规模统一多视图几何估计。<u>后续模型扩展了这一范式</u>，引入了多视图输入、视频输入、鲁棒对应建模、相机参数注入以及大规模 SfM 和 SLAM 应用，并通过大规模训练、多阶段架构和设计冗余将精度推至新高度。<u>相比之下，我们专注于围绕单一简单 Transformer 构建的最小建模策略</u>。

### Monocular depth estimation（单目深度估计）
<u>早期单目深度估计方法</u>依赖于在单域数据集上的全监督学习，这些方法通常会产生专门针对室内房间 _[75]_ 或户外驾驶场景 _[26]_ 的模型。这些早期的深度模型<u>在其训练域内取得了良好的准确性，但在泛化到新环境时表现不佳</u>，突显了跨域深度预测的挑战。

> _[75] _**Indoor Segmentation and Support Inference from RGBD Images（ECCV 2012）**：提出 NYUv2 <u>室内 RGB-D 数据集</u>，并引入室内场景分割与结构理解基准；该数据集成为早期单目深度估计与室内感知任务的主要训练与评测平台，但模型往往只能在室内域内取得最佳性能。
>
> _[26] _**Vision Meets Robotics: The KITTI Dataset（IJRR 2013）**：发布<u>大规模自动驾驶场景的多模态数据集（RGB、激光雷达、里程计）</u>，为户外深度估计、立体视觉和 SLAM 研究提供标准基准，但模型容易过拟合于特定的户外驾驶分布，泛化到其他场景存在挑战。
>

<u>现代通用方法 体现了</u>_<u>这一趋势</u>_，通过利用大规模多数据集训练和先进的架构，如 Vision Transformer_ [67]_ 或 DiT _[64]_。它们在数百万张图像上进行训练，学习广泛的视觉线索，并结合诸如仿射不变深度归一化等技术。<u>相比之下，我们的方法主要设计用于统一的视觉几何估计任务，但仍展示了具有竞争力的单目深度性能</u>。

> _这一趋势_：这里是指 模型不局限于训练数据，提升模型的泛化能力 的研究。
>

> _[67]_ **Vision Transformers for Dense Prediction（ICCV 2021）**：将 Vision Transformer 扩展到像素级任务，通过密集注意力与特征重建机制，显著提升在语义、深度等稠密任务中的性能，奠定后续 ViT-based 密集预测模型的基础。
>
> _[64]_ **Scalable Diffusion Models with Transformers（ICCV 2023）**：将 Transformer 引入扩散模型架构，提升训练与推理的可扩展性，为后续将 DiT 用于深度估计等密集预测任务提供基础架构。
>

### Feed-Forward Novel View Synthesis（前馈新视角合成）
_<u>新视角合成 (NVS</u>_<u>) 长期以来一直是计算机视觉和图形学中的核心问题</u>，并且随着神经渲染的兴起，其关注度也显著提升。<u>一个特别有前景的方向是</u>_<u>前馈新视角合成（Feed-Forward NVS）</u>_<u>，它通过图像到 3D 网络的一次单次传递即可生成 3D 表示，避免了繁琐的逐场景优化</u>。

 早期方法大多采用 NeRF 作为底层 3D 表示，但由于 3DGS 具有显式结构和实时渲染能力，<u>近期研究已逐渐转向基于 3DGS 的方法</u>。<u>代表性工作通过引入</u>**<u>几何先验</u>**（如极线注意力 _[11]_、代价体_ [13]_ 和深度先验_ [107]_）<u>来提升图像到 3D 网络的能力。更近期的工作将</u>**<u>多视图几何类基础模型</u>**<u> </u>_<u>[85, 91, 96, 110]</u>_<u> 整合到前馈框架中，以提升建模能力，尤其是在</u>**<u>无姿态</u>**<u>设置中 </u>_<u>[41, 79, 116]</u>_。然而，这类方法往往依赖某一个特定的基础模型进行评估。

在本文中，我们系统地基准比较不同几何基础模型对 NVS 的贡献，并提出更优的利用策略，<u>使得前馈式 3DGS 能够同时处理有姿态与无姿态输入、可变数量的视图以及任意分辨率的场景</u>。  

> _新视角合成 (NVS)_：旨在**从有限的输入视角生成新的未观测视角图像**。传统方法依赖逐场景优化（例如 NeRF），渲染速度慢且泛化能力弱。近年来，随着几何基础模型与 3D 高斯表示的发展，出现了_前馈新视角合成（Feed-Forward NVS）_，即通过单次网络前向传播直接获得场景的 3D 表示或目标视角图像，从而实现高效、可泛化、可实时的新视角生成。  
>

> _[11]_ **PixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction（CVPR 2024）**：<u>提出利用 极线注意力（epipolar attention）构建两视图之间的几何约束</u>，从而实现可泛化、可扩展的两视图 3DGS 重建，是前馈 3DGS 的早期代表作之一。
>
> _[13]_ **MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images（ECCV 2024）**：<u>构建 基于代价体（cost volume）的特征融合方式，使稀疏多视图输入能够高效生成 3D 高斯</u>，实现速度与质量兼顾的可泛化场景重建。
>
> _[107]_ **MURF: Multi-Baseline Radiance Fields（CVPR 2024）**：<u>提出利用 多基线视图和深度先验（depth prior）构建鲁棒的稠密场景表示</u>，大幅提高在稀疏视图条件下的几何一致性，是基于深度先验约束的典型方法。
>
> _[85]_ **MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views in 2 Seconds（CVPR 2025）**：<u>基于 DUSt3R 的多视图几何基础模型</u>，构建可直接从稀疏图像输入进行单阶段重建的系统，显著提高推理速度，可在数秒内完成场景恢复。
>
> _[91]_ **VGGT: Visual Geometry Grounded Transformer（CVPR 2025）**：提出以几何为核心的 Transformer 模型，作为基础模型学习可泛化的视觉几何表征，可用于匹配、深度估计和相机姿态预测等任务。
>
> _[96]_ **DUSt3R: Geometric 3D Vision Made Easy（CVPR 2024）**：提出一种高度泛化的几何基础模型，可从图像对直接预测密集深度与相对姿态，并可作为强大的几何 backbone 用于各种 3D 重建任务。
>
> _[110]_ **Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass（CVPR 2025）**：<u>构建可处理上千张图像的前馈几何基础模型，实现大规模场景的快速重建</u>，将前馈 NVS 推向真正的“多图大规模重建”方向。
>
> [41] **AnySplat: Feed-Forward 3D Gaussian Splatting from Unconstrained Views（TOG 2025）**：提出<u>无需结构化视角约束的前馈 3DGS 框架，能够从任意拍摄条件（unconstrained views）直接重建场景</u>，提高泛化性与适应性。
>
> [79] **Splatt3r: Zero-Shot Gaussian Splatting from Uncalibrated Image Pairs（arXiv 2024）**：<u>实现无需相机姿态的零样本两视图 3DGS 重建</u>，通过学习到的几何一致性约束弥补姿态缺失的难题。
>
> [116] **No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images（ICLR 2024）**：提出一种极简方法，<u>在完全无姿态的稀疏多视图设置下仍能重建 3DGS</u>，实现“无 pose 输入”的简洁前馈几何建模。
>

## 创新点归纳
+ Depth Anything 3（DA3）模型能够根据任意数量的视觉输入预测空间一致的几何形状，<u>无论是否已知相机姿态</u>；
+ <u>仅需一个简单的 Transformer 架构（例如原始的 DINOv2 编码器）即可作为骨干网络</u>，无需进行架构上的特殊设计；
+ <u>利用单一的深度光（Depth + Ray）预测目标避免了复杂的多任务学习</u>。

## DepthAnything 3
### Formulation（公式化）
我们将输入表示为 ![image](https://cdn.nlark.com/yuque/__latex/a27b790108fc56246d08c122f03626c2.svg)，其中每个![image](https://cdn.nlark.com/yuque/__latex/aa63a8cd33ab543d97fb54158889e4dd.svg)都对应一个图像。对于![image](https://cdn.nlark.com/yuque/__latex/27d0859b6bf531c6b637f910ddc84c59.svg)，这是一个单目图像，而对于![image](https://cdn.nlark.com/yuque/__latex/9f13001bdc16af38029aadcb793517a2.svg)，它代表视频或多视角集合。每个图像都有深度![image](https://cdn.nlark.com/yuque/__latex/66e29c624c5e147bd19393230aa454ca.svg)，相机外参 ![image](https://cdn.nlark.com/yuque/__latex/4163cfc659bd0153bb9079867bc14bca.svg)，以及内参![image](https://cdn.nlark.com/yuque/__latex/88c34c77f50e1b68fa9f25340ed4dd6e.svg)。相机也可以表示为![image](https://cdn.nlark.com/yuque/__latex/2816853b60c317e03f3d9e27963f4872.svg)，包含平移![image](https://cdn.nlark.com/yuque/__latex/0d552afdf977186d0e3cf650e9338673.svg)、旋转四元数![image](https://cdn.nlark.com/yuque/__latex/ab34fdc9bc4a5243e94e48d63e73d5f5.svg)，以及视场角参数![image](https://cdn.nlark.com/yuque/__latex/27f592aefd4102ef546e535aed6205a0.svg)。一个像素![image](https://cdn.nlark.com/yuque/__latex/06c868e6b8bdc696d501c5e6e60ceb59.svg)通过投影到一个 3D 点 ![image](https://cdn.nlark.com/yuque/__latex/56fea354068f62300d04b796151d76c0.svg)，

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765071902229-cc73e278-0c91-4502-842d-e662dac50cdf.png)

从而可以恢复其下的 3D 视觉空间。

#### Depth-ray representation（深度射线表示）
<font style="color:rgb(31, 35, 41);">由于</font>_<font style="color:rgb(31, 35, 41);">正交性约束</font>_<font style="color:rgb(31, 35, 41);">，预测一个有效的旋转矩阵</font>![image](https://cdn.nlark.com/yuque/__latex/842262380465be6f0f4f5401b1296ee4.svg)<font style="color:rgb(31, 35, 41);">是具有挑战性的。</font><u><font style="color:rgb(31, 35, 41);">为了避免这一问题，我们使用与输入图像和深度图对齐的每像素射线图来隐式表示相机位姿</font></u><font style="color:rgb(31, 35, 41);">。</font>

> _正交性约束_：![image](https://cdn.nlark.com/yuque/__latex/9e357f860e68ca2da246a67aa2c3a6cd.svg)，![image](https://cdn.nlark.com/yuque/__latex/4a7501381679aa51d16e23d8e71c6eb8.svg)
>

<font style="color:rgb(31, 35, 41);">对于每个像素</font>![image](https://cdn.nlark.com/yuque/__latex/8f4b562467978432828ee0e9ebaf90b3.svg)<font style="color:rgb(31, 35, 41);">，相机射线</font>![image](https://cdn.nlark.com/yuque/__latex/205ae0d74cd78a31c4ea1dffe4b8004e.svg)<font style="color:rgb(31, 35, 41);">由其原点</font>![image](https://cdn.nlark.com/yuque/__latex/8a2e8e9cd02c63935e3dfffad82223bc.svg)<font style="color:rgb(31, 35, 41);">和方向</font>![image](https://cdn.nlark.com/yuque/__latex/69b8382aeaff53a863b5f3fb1301b114.svg)<font style="color:rgb(31, 35, 41);">定义：</font>![image](https://cdn.nlark.com/yuque/__latex/18b3003876bb03d4a3df11f0a5231b95.svg)<font style="color:rgb(31, 35, 41);">。方向是通过将</font>![image](https://cdn.nlark.com/yuque/__latex/8f4b562467978432828ee0e9ebaf90b3.svg)<font style="color:rgb(31, 35, 41);">投影到相机坐标系并旋转到世界坐标系中获得的：</font>![image](https://cdn.nlark.com/yuque/__latex/4867e3306061e6b428fed009d7903b46.svg)<font style="color:rgb(31, 35, 41);">。</font>_<font style="color:rgb(31, 35, 41);">密集射线图</font>_![image](https://cdn.nlark.com/yuque/__latex/107c16905029e6df9f4bc7f6cd649416.svg)<font style="color:rgb(31, 35, 41);">存储了所有像素的这些参数。我们不归一化</font>![image](https://cdn.nlark.com/yuque/__latex/3e72b959ea7b0668f45ac6b6c67f4a23.svg)<font style="color:rgb(31, 35, 41);">，因此其幅度保留了投影的尺度。世界坐标系中的一个 3D 点可表示为</font>![image](https://cdn.nlark.com/yuque/__latex/6445f12f94226ca6420d91ee48cf2513.svg)<font style="color:rgb(31, 35, 41);">。</font><u><font style="color:rgb(31, 35, 41);">这种公式化通过逐元素操作将预测的深度和射线图结合，从而实现一致的点云生成</font></u><font style="color:rgb(31, 35, 41);">。</font>

> _<font style="color:rgb(31, 35, 41);">密集射线图</font>_<font style="color:rgb(31, 35, 41);">：M 的形状 (H, W, t + d)，具有逐像素的原点和方向信息。</font>
>
> 这公式与之前的投影公式基本一致。
>

#### Deriving Camera Parameters from the Ray Map（从射线图中推导相机参数）
给定一个输入图像![image](https://cdn.nlark.com/yuque/__latex/84cef5e78b7324f1cd07ce670354c8bc.svg)，对应的射线图表示为![image](https://cdn.nlark.com/yuque/__latex/b2091edb0bafe461e971c556087e6a8f.svg)。该图包含每个像素的射线原点，存储在前三个通道中(![image](https://cdn.nlark.com/yuque/__latex/c62e43c8793d2ab01f19d6bca7c517bc.svg))，以及射线方向，存储在后三个通道中(![image](https://cdn.nlark.com/yuque/__latex/414708c9021b42d136d5c5cc2e880ac4.svg))。

首先，<u>通过平均每个像素的射线原点向量来估计相机中心</u>![image](https://cdn.nlark.com/yuque/__latex/f19201de18590e220d6288dfcadf6b6e.svg)：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765072936034-46ba7711-181d-4360-a187-b4963fe15b7d.png)

> 理论上所有像素的原点都相同，但是存在噪声影响，所以需要进行平均。
>

```python
# Depth-Anything-3-main\src\depth_anything_3\utils\ray_utils.py L435-L504

# 函数输入：
#   - camray: 相机光线张量 (B, S, num_patches_y, num_patches_x, 6)
#   - confidence: 置信度张量 (B, S, num_patches_y, num_patches_x)
#   - reproj_threshold: 重投影误差阈值
#   - training: 是否为训练模式
# 函数作用：将相机光线转换为相机参数（旋转、平移、焦距、主点）
# 函数输出：
#   - R: 旋转矩阵 (B, S, 3, 3)
#   - T: 平移向量 (B, S, 3)
#   - focal_lengths: 焦距 (B, S, 2)
#   - principal_points: 主点 (B, S, 2)
def camray_to_caminfo(camray, confidence=None, reproj_threshold=0.2, training=False):
    """
    Args:
        camray: (B, S, num_patches_y, num_patches_x, 6)
        confidence: (B, S, num_patches_y, num_patches_x)
    Returns:
        R: (B, S, 3, 3)
        T: (B, S, 3)
        focal_lengths: (B, S, 2)
        principal_points: (B, S, 2)
    """
    # 如果未提供置信度，则创建全1的置信度张量
    if confidence is None:
        confidence = torch.ones_like(camray[:, :, :, :, 0])
    # 获取形状信息
    B, S, num_patches_y, num_patches_x, _ = camray.shape
    # 创建单位内参矩阵，假设图像宽高为2.0
    I_K = torch.eye(3, dtype=camray.dtype, device=camray.device)
    # 设置主点x坐标
    I_K[0, 2] = 1.0
    # 设置主点y坐标
    I_K[1, 2] = 1.0
    # 扩展单位内参矩阵以匹配camray的批次
    I_K = I_K.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)

    # 创建单位深度的相机平面
    cam_plane_depth = torch.ones(
        B, S, num_patches_y, num_patches_x, 1, dtype=camray.dtype, device=camray.device
    )
    # 使用单位内参矩阵对深度进行反投影，得匰3D点
    I_cam_plane_unproj = unproject_depth(
        cam_plane_depth,
        I_K,
        c2w=None,
        ixt_normalized=True,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )  # (B, S, num_patches_y, num_patches_x, 3)

    # 将camray展平为2D形状
    camray = camray.flatten(0, 1).flatten(1, 2)  # (B*S, num_patches_y*num_patches_x, 6)
    # 将反投影点展平为2D形状
    I_cam_plane_unproj = I_cam_plane_unproj.flatten(0, 1).flatten(
        1, 2
    )  # (B*S, num_patches_y*num_patches_x, 3)
    # 将置信度展平为2D形状
    confidence = confidence.flatten(0, 1).flatten(1, 2)  # (B*S, num_patches_y*num_patches_x)
    
    # 计算最优旋转以对齐光线
    # 获取点的数量
    N = camray.shape[-2]
    # 获取设备类型
    device = camray.device
    # 获取RANSAC参数
    n_iter, num_sample_for_ransac, n_sample, rand_sample_iters_idx = get_params_for_ransac(N, device)
    
    # 如果处于训练模式，需要分离梯度
    if training:
        camray = camray.clone().detach()
        I_cam_plane_unproj = I_cam_plane_unproj.clone().detach()
        confidence = confidence.clone().detach()
    # 使用批处理计算最优旋转和内参
    R, focal_lengths, principal_points = compute_optimal_rotation_intrinsics_batch(
        I_cam_plane_unproj,
        camray[:, :, :3],
        reproj_threshold=reproj_threshold,
        weights=confidence,
        n_sample = n_sample,
        n_iter=n_iter,
        num_sample_for_ransac=num_sample_for_ransac,
        rand_sample_iters_idx=rand_sample_iters_idx,
    )

    # 计算加权平均平移向量（使用camray的后3个维度）<- 公式 (1)
    T = torch.sum(camray[:, :, 3:] * confidence.unsqueeze(-1), dim=1) / torch.sum(
        confidence, dim=-1, keepdim=True
    )

    # 将旋转矩阵reshape回原始形状
    R = R.reshape(B, S, 3, 3)
    # 将平移向量reshape回原始形状
    T = T.reshape(B, S, 3)
    # 将焦跜reshape回原始形状
    focal_lengths = focal_lengths.reshape(B, S, 2)
    # 将主点reshape回原始形状
    principal_points = principal_points.reshape(B, S, 2)

    # 返回旋转矩阵、平移向量、焦距的倒数、调整后的主点
    return R, T, 1.0 / focal_lengths, principal_points + 1.0
```

```python
# Depth-Anything-3-main\src\depth_anything_3\utils\ray_utils.py L506-L523

# 函数输入：
#   - camray: 相机光线张量 (B, S, num_patches_y, num_patches_x, 6)
#   - conf: 置信度张量 (B, S, num_patches_y, num_patches_x, 1)
#   - patch_size_y: 块的y方向大小
#   - patch_size_x: 块的x方向大小
#   - training: 是否为训练模式
# 函数作用：从相机光线提取相机外参矩阵、焦距和主点
# 函数输出：
#   - pred_extrinsic: 预测外参矩阵 (B, S, 4, 4)
#   - pred_focal_lengths: 预测焦距 (B, S, 2)，对应 fx，fy
#   - pred_principal_points: 预测主点 (B, S, 2)，对应 cx，cy
def get_extrinsic_from_camray(camray, conf, patch_size_y, patch_size_x, training=False):
    # 将相机光线转换为相机参数
    pred_R, pred_T, pred_focal_lengths, pred_principal_points = camray_to_caminfo(
        camray, confidence=conf.squeeze(-1), training=training
    )

    # 构造外参矩阵：合并旋转矩阵和平移向量(w2c)
    pred_extrinsic = torch.cat( 
        [
            # 合并R和T为[R|T]形式
            torch.cat([pred_R, pred_T.unsqueeze(-1)], dim=-1),
            # 添加底部行[0, 0, 0, 1]
            repeat(
                torch.tensor([0, 0, 0, 1], dtype=pred_R.dtype, device=pred_R.device),
                "c -> b s 1 c",
                b=pred_R.shape[0],
                s=pred_R.shape[1],
            ),
        ],
        dim=-2,
    )  # B, S, 4, 4
    # 返回外参矩阵、焦距和主点
    return pred_extrinsic, pred_focal_lengths, pred_principal_points
```

```python
# DepthAnything3/src/depth_anything_3/model/da3.py L181-L203
    def _process_ray_pose_estimation(
        self, output: Dict[str, torch.Tensor], height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """
        输入:
            output - 包含光线预测的输出字典
            height - 图像高度
            width - 图像宽度
        作用: 如果存在光线位姿解码器，从光线预测计算相机外参和内参
        输出: 更新后的输出字典，包含相机外参和内参
        """
        # 如果输出中包含光线信息和光线置信度
        if "ray" in output and "ray_conf" in output:
            # 从相机光线获取相机外参、焦距和主点
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],
                output.ray.shape[-2],
            )
            # 将世界坐标系到相机坐标系 (w2c) 转换为相机到世界 (c2w)
            pred_extrinsic = affine_inverse(pred_extrinsic) # w2c -> c2w
            # 只保留旋转和平移部分 (3x4)
            pred_extrinsic = pred_extrinsic[:, :, :3, :]
            # 创建单位矩阵作为内参矩阵的基础
            pred_intrinsic = torch.eye(3, 3)[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone().to(pred_extrinsic.device)
            # 设置 x 轴的焦距（从归一化的焦距转换为像素单位）
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
            # 设置 y 轴的焦距
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
            # 设置 x 轴的主点坐标
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
            # 设置 y 轴的主点坐标
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
            # 删除中间的光线信息，不再需要
            del output.ray
            del output.ray_conf
            # 将预测的外参和内参存入输出字典
            output.extrinsics = pred_extrinsic
            output.intrinsics = pred_intrinsic
        return output
```

为了估计旋转![image](https://cdn.nlark.com/yuque/__latex/1227f701376733061841f20c2cacfce0.svg)和内参![image](https://cdn.nlark.com/yuque/__latex/fbd2339328b7567cf8beb0c5239de4f5.svg)，我们将问题表述为寻找一个单应性变换![image](https://cdn.nlark.com/yuque/__latex/196d2c14d39338b7881a690c016f3fbf.svg)。我们首先定义一个具有<u>单位内参矩阵</u>的“规范”相机，![image](https://cdn.nlark.com/yuque/__latex/4e2dc619670a140af135be5229f8eb39.svg)。对于给定的像素![image](https://cdn.nlark.com/yuque/__latex/8f4b562467978432828ee0e9ebaf90b3.svg)，其在该规范相机坐标系中的对应射线方向为![image](https://cdn.nlark.com/yuque/__latex/776536491ff39b48304b0d454f619470.svg)。从该规范射线![image](https://cdn.nlark.com/yuque/__latex/462b2d8d823979975294cfb5816d7dbc.svg)到目标相机坐标系中的射线方向![image](https://cdn.nlark.com/yuque/__latex/a938820a907b4e8da9f9b08acedb1c04.svg)的变换由![image](https://cdn.nlark.com/yuque/__latex/02c0c95913fa4ac32b5855212e1440f6.svg)给出。这建立了两个射线集合之间的直接单应性关系![image](https://cdn.nlark.com/yuque/__latex/4611c7589e8304be04f623789368164e.svg)。然后我们可以<u>通过最小化变换后的规范射线与一组预计算的目标射线</u>![image](https://cdn.nlark.com/yuque/__latex/3a620490527c495bffcc5560a7e51bb0.svg)<u>之间的几何误差来求解该单应性变换</u>。这导致了以下_优化问题_：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765073063691-702f64ac-6d3b-4a71-b228-7696afd95350.png)

这是一个标准的最小二乘问题，可以使用直接线性变换（DLT）算法高效求解。一旦找到最优的单应性变换![image](https://cdn.nlark.com/yuque/__latex/0fea4c21b87de205561c16178c500cd9.svg)，我们就可以恢复相机参数。由于内参矩阵![image](https://cdn.nlark.com/yuque/__latex/fbd2339328b7567cf8beb0c5239de4f5.svg)是上三角矩阵，而旋转矩阵![image](https://cdn.nlark.com/yuque/__latex/1227f701376733061841f20c2cacfce0.svg)是正交归一化的，我们可以使用 RQ 分解唯一地分解![image](https://cdn.nlark.com/yuque/__latex/0fea4c21b87de205561c16178c500cd9.svg)，从而得到![image](https://cdn.nlark.com/yuque/__latex/eb91df83b1dfb7561fd744e520ec5721.svg)。

> _优化问题_：射线方向![image](https://cdn.nlark.com/yuque/__latex/d689a7955bede1a270cf6d23c90b6ca2.svg)理应与射线图![image](https://cdn.nlark.com/yuque/__latex/7679762b171cdd352f92dead5cefd69f.svg)中对应像素位置射线方向相同，那么两个方向相同的向量之间叉积应该为 0，从而构建了该优化问题。
>

#### Minimal prediction targets（最小化预测目标）
近期工作旨在构建统一模型以处理多样化的 3D 任务，通常使用多任务学习，其目标各不相同——例如，仅使用点图，或位姿、局部/全局点图和深度的冗余组合。<u>虽然点图不足以保证一致性，但冗余目标可以提高姿态精度，但常常引入纠缠现象，从而损害其精度</u>。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765154802867-53aca44b-3463-4f39-b5ad-bbbdf81bf1cc.png)

**表 6. 预测‐目标组合的消融实验：**请注意,本表中的所有实验均不包含相机条件标记。The best 和第二好的结果被突出显示。

> pcd 为点云信息，cam 为相机内外参。
>

与之相比，我们的实验(_表 6 _)表明，<u>深度射线表示形成了一组最小但充分的目标集合，用于捕捉场景结构和相机运动，优于点图或其他更复杂的输出</u>。然而，在推理过程中从射线图中恢复相机姿态计算成本较高。 我们通过添加一个轻量级的相机头来解决这个问题。该 Transformer 作用于相机标记，以预测视野(f)，旋转作为四元数(q)，以及平移(t)。由于它每视图仅处理一个标记，因此增加的成本可以忽略不计。

### Architecture（架构）
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765073539620-a4e4e21f-9358-4e3b-a5a0-2c2f8df919bd.png)

**图 2. DA 3 的流程图：**Depth Anything 3 采用了一个单个Transformer(_vanilla DINOv2模型_)，没有任何架构上的修改。为实现跨视图推理，引入了输入自适应跨视图自注意力机制。使用了一个 Dual‐DPT 头，从视觉标记中预测深度和射线图。如果有可用的相机参数，它们会被编码为相机标记并与补丁标记连接，参与所有的注意力操作。

我们详细描述 Depth Anything 3 的架构，如_图 2_ 所示。该网络由三个主要组件组成：一个单个 Transformer 模型作为主干网络，一个可选的相机编码器用于已知位姿情况，以及一个 Dual‐DPT 头用于生成预测。

> _vanilla DINOv2 模型_：未在结构上做任何改造的 DINOv2 模型，此处 DINOv2 用于获取图像的二维语义特征，随后 Dual‐DPT Head 基于此特征来获取深度图和射线图信息。
>

#### Single transformer backbone（单个 Transformer 主干网络）
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765156394057-07f839c7-c5ab-4090-ae62-dadcc737f005.png)

**表 7. 消融研究**： 我们评估了三种具有可比模型规模 (a‐c) 的架构设计，双DPT头(d)、教师标签监督 (e) 以及姿态条件模块 (f‐g) 的效果。 最佳和第二佳方法被突出显示。带有"*"标记的方法使用真实值姿态融合进行评估。

> 这里只看 a，b，c，其中 a 为原文提出的架构，b 为 VGGT 架构，其采用规模较小的 ViT-B 作为主干网络，c 为将所以层均改为交替的跨视图和内视图注意力，即![image](https://cdn.nlark.com/yuque/__latex/3026969af686dc29f15ebcad43645fdc.svg)。
>

我们使用一个视觉 Transformer，包含![image](https://cdn.nlark.com/yuque/__latex/5e588dfc5879cf1d6455640acebad04c.svg)个块，预训练于大规模单目图像语料库（例如，DINOv2 _[61]_）。通过输入自适应自注意力，无需架构更改即可启用跨视图推理，该自注意力通过重新排列输入标记实现。<u>我们将 Transformer 分为两个组，大小分别为</u>![image](https://cdn.nlark.com/yuque/__latex/2ee8fd0432b23b4c1852226e6dac95b5.svg)<u>和</u>![image](https://cdn.nlark.com/yuque/__latex/e85489e80122c44df0eb168ed7bf2642.svg)<u>。前 </u>![image](https://cdn.nlark.com/yuque/__latex/b709be1a2cb0bb57e76bea8d48035bf2.svg)<u>层在每张图像内应用</u>_<u>自注意力（self-）</u>_<u>，而后续</u>![image](https://cdn.nlark.com/yuque/__latex/e85489e80122c44df0eb168ed7bf2642.svg)<u>层则交替进行</u>_<u>跨视图（cross-view）和内视图注意力（within-view）</u>_<u>，通过张量重新排列对所有标记进行联合操作。</u>在实践中，我们设置 ![image](https://cdn.nlark.com/yuque/__latex/7e51b958a5251d3ff113241cfddbbb46.svg)，其中 ![image](https://cdn.nlark.com/yuque/__latex/e42ad6e55d7fd55c1387410f011a8911.svg)。<u>如我们在</u>_<u>表 7 </u>_<u>中的消融研究所示，这种配置相比其他安排在性能和效率之间提供了最佳的权衡</u>。这种设计是输入自适应的：对于单张图像，模型自然退化为单目深度估计，无需额外成本。

> _自注意力、跨视图注意力 _和_内视图注意力_：每个 token 与所有 token 建立关系、不同图像之间 tokens 的匹配与融合 和 同一张图像内部 tokens 之间的关系。
>

> _**[61]**_** DINOv2: Learning Robust Visual Features without Supervision（arXiv 2023）**：提出一种基于 Vision Transformer 的大规模自监督视觉预训练方法，大幅提升跨任务与跨数据集迁移能力。通过改进特征蒸馏、数据增广与训练策略，DINOv2 构建出具有更强语义一致性和更稳定几何结构的表征，为多视图推理、场景重建及单目几何任务提供了更高质量的视觉特征基础。  
>

#### Camera condition injection（相机条件注入）
<u>为了无缝处理有姿态和无姿态输入，我们为每个视图添加一个相机标记</u>![image](https://cdn.nlark.com/yuque/__latex/7b3f8a3284880c1d37db4bf6fe98c6ee.svg)。如果相机参数![image](https://cdn.nlark.com/yuque/__latex/c5b36618982faf6c382fb9e1198f6002.svg)可用，该标记通过一个轻量级 MLP![image](https://cdn.nlark.com/yuque/__latex/c278dc1d4c54d7e13daf1143fe838d34.svg) 获得：![image](https://cdn.nlark.com/yuque/__latex/c927627a30e03156c2c8d1e73efd4f95.svg)。否则，使用一个共享的可学习标记![image](https://cdn.nlark.com/yuque/__latex/03a370c053f9bc4c0648ab69d3f87d9b.svg)。这些相机标记与补丁标记连接后，参与所有注意力操作，提供显式的几何上下文或一致的可学习占位符。

```python
# DepthAnything3/src/depth_anything_3/model/da3.py L100-L134
    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:
        """
        输入:
            x - 输入图像 (B, N, 3, H, W)，B=批次大小，N=视图数量
            extrinsics - 相机外参 (B, N, 4, 4)，可选
            intrinsics - 相机内参 (B, N, 3, 3)，可选
            export_feat_layers - 需要导出的特征层索引列表
            infer_gs - 是否启用高斯散点 (Gaussian Splatting) 分支
            use_ray_pose - 是否使用基于光线的位姿估计
            ref_view_strategy - 参考视图选择策略
        作用: 网络的前向传播，执行深度估计、相机位姿估计和可选的3DGS预测
        输出: 包含预测结果和辅助特征的字典
            - depth: 预测深度图 (B, H, W)
            - depth_conf: 深度置信度 (B, H, W)
            - extrinsics: 相机外参 (B, N, 4, 4)
            - intrinsics: 相机内参 (B, N, 3, 3)
            - gaussians: 3D高斯散点（世界坐标系）
            - aux: 指定层的辅助特征
        """
        # 如果提供了相机外参，使用相机编码器生成相机 token
        if extrinsics is not None:
            # 关闭混合精度训练，确保计算精度
            with torch.autocast(device_type=x.device.type, enabled=False):
                # 将相机参数编码为 token
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            # 如果没有提供外参，cam_token 为 None
            cam_token = None
        
        # 使用骨干网络提取特征
        # feats: 主要特征，aux_feats: 辅助特征（用于导出）
        feats, aux_feats = self.backbone(
            x, cam_token=cam_token, export_feat_layers=export_feat_layers, ref_view_strategy=ref_view_strategy
        )
```

#### Dual-DPT head
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765075746483-643de9ca-4436-4320-b63c-e308e9cd5111.png)

**图 3.Dual‐DPT 头：**两个分支共享重组模块以实现更好的输出对齐。

<u>在最终预测阶段，我们提出了一种新颖的 Dual‐DPT 头，联合生成密集深度和射线值</u>。如_表 6 _所示，这种设计既强大又高效。<u>给定主干网络的一组特征，Dual‐DPT 头首先通过一组共享的重组模块处理这些特征。随后，处理后的特征通过两组不同的融合层进行融合：一组用于深度分支，另一组用于射线分支。最后，两个独立的输出层生成最终的深度和射线图预测</u>。这种架构确保两个分支在相同的处理特征集上操作，仅在最终融合阶段有所不同。这样的设计促进了两个预测任务之间的强交互，同时避免了冗余的中间表示。

### Training（训练）
> ![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765176243487-3bb106dc-144c-469b-bc89-2667ce444fe3.png)
>
> 训练部分代码未公布，此处未对相应公式进行代码注释。
>

#### Teacher-student learning paradigm（教师‐学生学习范式）
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765075930978-30911d52-4a51-4b0a-bedc-0dfa1c15db57.png)

**图 4. 质量差的真实世界数据集**：我们展示了一些质量差的真实世界数据集的例子。

<u>我们的训练数据来源于多种来源，包括现实世界深度捕捉、3D重建和合成数据集</u>。<u>现实世界深度往往具有噪声且不完整（</u>_<u>图 4 </u>_<u>），限制了其监督价值。为缓解这一问题，我们</u>**<u>仅使用合成数据训练一个单目相对深度估计的“教师”模型</u>**<u>，以生成高质量的伪标签</u>。这些伪深度图通过 RANSAC 最小二乘法与原始稀疏或噪声的地面真实值对齐，增强标签的细节和完整性，同时保持几何精度。<u>我们将此模型称为 Depth‐Anything‐3‐Teacher</u>，训练于一个覆盖室内、户外、以物体为中心以及多样真实场景的大型合成语料库，以捕捉精细几何。

#### Training objectives（训练目标）
<u>我们的模型</u>![image](https://cdn.nlark.com/yuque/__latex/7d6df54920c1174670cd63658952561f.svg)<u>将输入</u>![image](https://cdn.nlark.com/yuque/__latex/9291e3e5dc14a700640b9bd992971d7b.svg)<u>映射到一组输出，包括：深度图</u>![image](https://cdn.nlark.com/yuque/__latex/a0d9047a91bfb0b89f8706963fe8eded.svg)<u>、射线图</u>![image](https://cdn.nlark.com/yuque/__latex/736fb82f533ec7a5df6463059b0e57aa.svg)<u> 和一个可选的相机姿态</u>![image](https://cdn.nlark.com/yuque/__latex/d382060d2a30bb15e7d5feccc8165991.svg)：![image](https://cdn.nlark.com/yuque/__latex/705b19d7808c5ea68383ed0669412a29.svg)。灰色表示![image](https://cdn.nlark.com/yuque/__latex/d382060d2a30bb15e7d5feccc8165991.svg)是一个可选输出，主要出于实际便利性而包含。在损失计算之前，所有真实信号都通过一个公共的尺度因子进行归一化。该尺度定义为有效重投影点图的均值 ![image](https://cdn.nlark.com/yuque/__latex/ea2f653770675d7cdd2c6a2618b44707.svg) norm，这一步确保了不同模态之间的一致性幅度，并稳定了训练过程。整体的训练目标定义为几个项的加权和：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765076498698-41cfacaf-88e9-49e6-9a1a-93b7966e20eb.png)

其中![image](https://cdn.nlark.com/yuque/__latex/867382bb8d05db9b4a074d35a5487930.svg)表示深度![image](https://cdn.nlark.com/yuque/__latex/12b8ebb43a1482ab7cc290db8293a619.svg)的置信度。所有损失项都基于![image](https://cdn.nlark.com/yuque/__latex/f814378eceab0083189bbdb1fa0c2b65.svg)范数，权重设置为![image](https://cdn.nlark.com/yuque/__latex/176f0d014227ce48fcfda6f1d9b4b35d.svg)和![image](https://cdn.nlark.com/yuque/__latex/662e5db38a02eb7d9e4a6453bfbb5717.svg)。梯度损失![image](https://cdn.nlark.com/yuque/__latex/ad9b61bf38cb666a7e6e45740ee180ab.svg)对深度梯度进行惩罚：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765076534009-00f32bae-765f-4b6b-b69b-45578d6102a2.png)

其中 ![image](https://cdn.nlark.com/yuque/__latex/03bb077d37c907a4a46f7828d160f988.svg) 和 ![image](https://cdn.nlark.com/yuque/__latex/978243f9028b79e1d2adf6fbcd28d3fc.svg) 是水平和垂直的有限差分算子。这种损失函数在保持边缘锐利的同时，确保平面区域的平滑性。在实践中，我们设置 ![image](https://cdn.nlark.com/yuque/__latex/176f0d014227ce48fcfda6f1d9b4b35d.svg) 和 ![image](https://cdn.nlark.com/yuque/__latex/662e5db38a02eb7d9e4a6453bfbb5717.svg)。

### Implementation Details（实现细节）
#### Traing datasets（训练数据集）
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765157311906-68f5154f-8a48-4cbd-a687-c2bbdbf306da.png)

**表 1**. Depth Anything 3 中使用的数据集,包括场景数量、数据类型。

> Synthetic：合成数据；Colmap：SfM+MVS 重建数据。
>

我们在_表 1_ 中提供了我们的训练数据集。 注意，<u>对于训练和测试之间可能存在重叠的数据集（如 ScanNet++），我们确保在场景级别进行严格分离，即训练和测试中的场景是互斥的</u>。

#### Training details（训练细节）
我们在 128 台 H100 GPU 上对模型进行训练，共进行 200000 步，使用 8000 步的预热阶段和峰值学习率![image](https://cdn.nlark.com/yuque/__latex/536efc99543387b756507ad78d31900e.svg)。基础分辨率为 504 × 504，它可以被 2、3、4、6、9 和 14 整除，使其更兼容常见的照片宽高比，如 2:3、3:4 和 9:16。训练图像分辨率是从 504 × 504，504 × 378，504 × 336，504 × 280，336 × 504，896 × 504，756 × 504，672 × 504 中随机采样的。批量大小动态调整以保持每步的标记数大致恒定。监督从真实深度过渡到教师模型标签是在 120000 步时进行的。姿态条件化在训练过程中以 0.2 的概率随机激活。

## Teacher-Student Learning
<u>由于真实世界数据集质量较差，因此我们仅使用合成数据训练教师模型以对真实世界数据进行监督</u>。<u>我们的教师模型被训练为一个单目相对深度预测器。</u>在推理或监督过程中，可以使用有噪声的真实深度来提供缩放和偏移参数，从而实现预测的相对深度与绝对深度测量的对齐。

### Constructing the Teacher Model（构建教师模型）
基于 DA2 的研究基础，我们在数据和表征等多个关键方面对该方法进行了扩展。<u>我们观察，扩大训练语料库可以明显提升深度估计性能，这证实了数据规模化的益处</u>。此外，尽管我们改进的深度表征在标准二维评估指标上可能未显示出显著的提升，但它能产生质量更好的三维点云，表现出更少的几何畸变和更逼真的场景结构。需要指出的是，<u>我们的教师网络主干与上述 DA3 框架直接对齐，仅由一个 DINOv2 视觉 Transformer 和一个 DPT 解码器组成——未引入任何专门的架构修改</u>。我们将在以下章节详细阐述完整的设计与实现细节。

#### Data scaling（数据缩放）
我们仅在合成数据上训练教师模型，以获得更精细的几何细节。<u>DA2 中使用的合成数据集相对有限。 在 DA3 中，我们大幅扩展了训练语料库</u>，包括：Hypersim [69]，TartanAir [98]，IRS [93]，vKITTI2 [9]，BlendedMVS [115]，SPRING [56]，MVSSynth [38]，UnrealStereo4K [123]，GTA‐SfM [92]，TauAgent [27]，KenBurns [60]，MatrixCity [50]，EDEN [47]， ReplicaGSO [82]，UrbanSyn [32]，PointOdyssey [127]，Structured3D [125]，Objaverse [17]，Trel‐ lis [104] 和OmniObject [102]。<u>该数据集涵盖了室内、户外、以物体为中心以及多样化的现实场景，从而提升了教师模型的泛化能力</u>。

#### Depth representation（深度表示）
<u>与 DA2 不同，DA2 预测的是尺度‐平移不变</u>**<u>视差</u>**<u>，而我们的教师输出的是尺度‐平移不变</u>**<u>深度</u>**。深度对于下游任务（如度量深度估计和多视角几何）更优，因为这些任务直接在深度空间中操作，而不是视差空间。<u>为了解决深度在近相机区域相比视差的敏感度降低问题，我们预测指数深度而不是线性深度，从而增强短距离的区分能力</u>。

#### Training objectives（训练目标）
<u>在几何监督方面，除了标准的深度梯度损失外，我们采用了文献 </u>_<u>[95]</u>_<u> 中提出的全局-局部损失下的 ROE 对齐方法。为了进一步优化局部几何特征，我们引入了距离加权的表面法向量损失</u>。对于每个中心像素，我们采样四个相邻点并计算未归一化的法向量![image](https://cdn.nlark.com/yuque/__latex/77db1de84c57b9099d7206fb2c0f1ca3.svg)。然后通过以下方式对这些法向量进行加权：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765077527649-43de6455-fc80-42f5-b718-c187226dc0a9.png)

这一权重会降低距离中心较远的邻点的贡献度，从而得到更接近真实局部表面法向量的平均法向量：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765077544408-3664fc55-ea3c-4f93-819a-71e2c3d8991f.png)

最终的法向量损失为：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765077557748-716246a4-a240-4a54-95d5-1a59fc965b92.png)

其中![image](https://cdn.nlark.com/yuque/__latex/201c332d65d99168e5a95c980d8c5e83.svg)表示法向量之间的角度误差。<u>在天空区域以及仅含物体的数据集的背景区域中，真实值是未定义的。为了避免这些区域降低深度预测的质量并便于下游使用，我们联合预测与深度输出对齐的天空掩码和物体掩码，并使用 MSE 损失进行监督</u>。整体训练目标为：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765181566680-8c4bf96c-c422-4c58-9b3e-378086590f9d.png)

其中![image](https://cdn.nlark.com/yuque/__latex/95a1f3cb18ed6d07ba86f0cd2ba4cf88.svg)。这里，![image](https://cdn.nlark.com/yuque/__latex/ad9b61bf38cb666a7e6e45740ee180ab.svg)、![image](https://cdn.nlark.com/yuque/__latex/6c9600f084ad01850a2407154b4a6a17.svg)、![image](https://cdn.nlark.com/yuque/__latex/8a536fde4d11ddb0d544ebfe4b03c966.svg) 和 ![image](https://cdn.nlark.com/yuque/__latex/9bb5a8ce9ce9427492c1ccfbbe48b3a7.svg) 分别表示梯度损失、全局-局部损失、天空掩码损失和物体掩码损失。

> _[95]_** MOGE: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision（CVPR 2025）**：提出 MOGE 框架，<u>通过设计全局–局部一致性的 ROE（Relative Ordinal Error）几何监督、改进的深度–法向量联合训练策略，以及开放域数据上的最优监督组合，使单目几何估计在无约束场景中获得显著提升</u>；在深度、法向、边界一致性等任务中均实现 SOTA，为开放域单目几何学习奠定了新的基准。  Teaching Depth Anything 3
>

### Teaching Depth Anything 3
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765077837813-a3c4e265-8470-4d8e-aace-2cfea7c36909.png)

**图 8. 教师标签监督对比：**使用教师生成的标签进行监督可以获得细节更丰富、结构更精细的深度图。

真实世界数据集对于相机位姿估计的泛化至关重要，但这类数据集很少提供干净的深度信息；监督信号通常是带噪声或稀疏的。<u>Depth Anything 3 Teacher 提供了高质量的相对深度</u>，我们通过鲁棒的 RANSAC 尺度-偏移流程，将其与带噪声的度量测量（如 COLMAP）对齐。令![image](https://cdn.nlark.com/yuque/__latex/93d4cf008ea922dc5336daa6100d6129.svg)表示教师模型的相对深度，![image](https://cdn.nlark.com/yuque/__latex/9dbc76bf179a997c5fcf48c57a8aa02a.svg)表示可用的稀疏深度（在域![image](https://cdn.nlark.com/yuque/__latex/ac952fb2338a5acc539486cbdcae059b.svg)上带有有效性掩码![image](https://cdn.nlark.com/yuque/__latex/26c5f209564f0161ac8f5b86b47db222.svg)）。<u>我们通过 RANSAC 最小二乘法估计尺度</u>![image](https://cdn.nlark.com/yuque/__latex/79ce3c7a71877c2ff01695e38ade43ca.svg)<u>和偏移</u>![image](https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg)，内点阈值设为残差中位数的平均绝对偏差：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765077782765-92cf8140-aa17-40e1-b0ac-9ce813f355d7.png)

对齐后的 ![image](https://cdn.nlark.com/yuque/__latex/a56b416625d82a9b86d5b2aea2361cf6.svg) 为 Depth Anything 3 提供了尺度一致且位姿-深度协调的监督信号，补充了我们的深度-射线联合目标，提升了真实世界场景下的泛化能力（如_图 8_ 所示）。

> 这里是对真实世界数据集通过 Teacher 模型重新预估深度后，将修改后的数据集用于 DA3 训练。
>

### Teaching Monocular Model
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765078014834-dcdbf4b9-79c0-4166-9a8a-5ac986f98790.png)

**表 10**

<u>我们还采用 teacher–student 范式</u>训练一个单目深度模型。我们遵循 DA2 框架，<u>在无标签图像上使用教师生成的伪标签训练单目学生模型</u>。<u>与 DA2 的关键区别在于预测目标：我们的学生模型预测深度图，而 DA2 预测视差</u>。我们进一步使用与教师相同的损失函数对学生的伪深度标签进行监督。单目模型还预测相对深度。仅使用无标签数据和教师监督进行训练，<u>它在标准单目深度基准测试中实现了最先进的性能，如</u>_<u>表 10</u>_。

### Teaching Metric Model
<u>接下来，我们证明我们的教师模型可用于训练具有清晰边界的</u>_<u>度量深度估计模型</u>_。在 Metric3D v2_ [37]_ 之后，<u>我们应用规范相机空间变换以解决由不同焦距引起的深度歧义</u>。<u>具体来说，我们使用比率 </u>![image](https://cdn.nlark.com/yuque/__latex/7510a0eb6dfb0bc9e0f364ce2c2a6c2d.svg)<u>对真实深度进行重缩放，其中</u>![image](https://cdn.nlark.com/yuque/__latex/de099e8dd7a71b0575cd990500f0cf5d.svg)<u>和</u>![image](https://cdn.nlark.com/yuque/__latex/18f3c2855f0e85a1ac2257f64d917144.svg)<u>分别表示 规范焦距 和 相机焦距</u>。为了确保细节清晰，我们采用教师模型的预测作为训练标签。我们将教师模型预测的深度的尺度和偏移与真实度量深度标签对齐以进行监督。

> _度量深度估计模型_：相较单目深度预测模型，其输出的尺度为真实尺度。
>

> _[37]_** **Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation（TPAMI 2024）：提出统一的单目几何基础模型** **Metric3D v2，<u>通过大规模跨场景训练和精心设计的几何一致性监督，实现对度量深度与表面法向的零样本高精度估计</u>；在室内、室外及跨域任务中均取得显著提升，为开放域单目几何感知提供了强泛化能力的通用框架。  
>

#### Training dataset（训练数据集）
我们在 14 个数据集上训练了我们的度量深度模型，包括 Taskonomy、DIML（Outdoor）、DDAD、Argoverse、Lyft、PandaSet、Waymo、ScanNet++、ARKitScenes、Map-free、DSEC、Driving Stereo 和 Cityscapes 数据集。对于立体数据集，我们利用 FoundationStereo _[100]_ 的预测作为训练标签。

> _[100]_** **FoundationStereo: Zero-Shot Stereo Matching（CVPR 2025）：<u>提出无需针对特定数据集微调的 零样本立体匹配基础模型 FoundationStereo</u>，通过结合大规模训练、鲁棒的几何一致性损失以及跨域特征对齐策略，实现对未见场景和新域的高精度视差估计；在大量真实和合成数据集上均展现强泛化能力，为立体感知任务提供了可直接作为监督信号的通用立体基础模型。  
>

#### Implementation Details（实现细节）
训练过程在很大程度上遵循单目教师模型的训练方式。所有图像均在基础分辨率为 504 的情况下进行训练，且具有不同的宽高比（1:1, 1:2, 16:9, 9:16, 3:4, 1:1.5, 1.5:1, 1:1.8）。我们采用 AdamW 优化器，并将编码器和解码器的学习率分别设置为5e-6 和 5e-5。我们应用随机旋转增强，训练图像以 90 或 270 度的概率进行旋转。我们设置规范焦距![image](https://cdn.nlark.com/yuque/__latex/4176cb07edc5accb7672918fa3289893.svg)为 300。我们使用教师模型的对齐预测作为监督。以 20% 的概率，我们使用原始真实标签进行训练。我们以批量大小为 64，训练 160K 次迭代。训练目标是深度损失![image](https://cdn.nlark.com/yuque/__latex/3bfa7e736eb8e0fa4ea8e1e26e27f6b1.svg)、![image](https://cdn.nlark.com/yuque/__latex/ad9b61bf38cb666a7e6e45740ee180ab.svg)和天空掩码损失![image](https://cdn.nlark.com/yuque/__latex/8a536fde4d11ddb0d544ebfe4b03c966.svg)的加权和。

## Application: Feed-Forward 3D Gaussian Splattings（应用:前馈 3D高斯点云生成）
### Pose-Conditioned Feed-Forward 3DGS（基于位姿的前馈 3DGS）
受人空间智能启发，我们认为一致的深度估计可以显著增强下游 3D 视觉任务。<u>我们选择前馈新视角合成（FF-NVS）作为演示任务</u>，因其在神经 3D 表示技术进步（即，我们选择 3D 高斯）的推动下受到越来越多的关注，并且与众多应用相关。<u>遵循最小建模策略，我们通过微调并添加 DPT 头（GS-DPT）来推断像素对齐的 3D 高斯</u>。

#### GS-DPT head
给定通过我们的单个 Transformer 主干网络（第 1.4.2 节）提取的每个视图的视觉标记，<u>GS-DPT 预测相机空间中的 3D高斯参数</u>![image](https://cdn.nlark.com/yuque/__latex/3ce22b94817e08deae4650ce6accfaaf.svg)。，其中![image](https://cdn.nlark.com/yuque/__latex/4b89e6d2334a1e829d1d44d8c842df5e.svg)分别表示第![image](https://cdn.nlark.com/yuque/__latex/2443fbcfeb7e85e1d62b6f5e4f27207e.svg)个 3D 高斯的不透明度、旋转四元数、尺度和 RGB 颜色。<u>其中，</u>![image](https://cdn.nlark.com/yuque/__latex/c302e028f05f811e7586b967f6dfa479.svg)<u>由置信度头预测，其余参数由主 GS-DPT 头部预测</u>。<u>估计的深度被反投影到世界坐标系中，以获得 3D 高斯的全局位置</u> ![image](https://cdn.nlark.com/yuque/__latex/2dce8b386fe75fc63b881392981bd3a2.svg)这些基础元素随后被光栅化，以从给定的相机姿态合成新颖视角。

#### Training objectives（训练目标）
NVS 模型通过两个训练目标进行微调，即在渲染的新颖视角上使用光度损失（即![image](https://cdn.nlark.com/yuque/__latex/00e529dfd2b9023126330ce1cee7ed02.svg)和![image](https://cdn.nlark.com/yuque/__latex/0963aeffa513210c1e2f1c749352c267.svg)）以及尺度 - 平移不变的深度损失![image](https://cdn.nlark.com/yuque/__latex/778016daebf7ecd8d5d0d37d3dcd3d41.svg)对观察视角的估计深度，遵循教师 - 学生学习范式。

### Pose-Adaptive Feed-Forward 3DGS（姿态自适应的前馈3DGS）
与上述旨在作为强前馈 3DGS 主干网络进行基准测试的位姿条件版本不同，<u>我们还提出了一种更适合野外评估的替代方案。该版本设计为使用相同的预训练权重无缝集成到 DA3 中，</u>**<u>从而实现带或不带相机姿态的新视角合成</u>**<u>，并适用于不同分辨率和输入视角数量的情况</u>。

#### Pose-adaptive formulation（位姿自适应公式化）
<u>我们采用一种位姿自适应设计</u>，而不是假设所有输入图像都是未校准的，接受 有位姿 和 无位姿 输入，从而得到一个灵活的框架，<u>无论是否有位姿都能工作</u>。实现这一目标需要两个设计选择: 1) 所有 3DGS 参数都在局部相机空间中预测。 2) 主干网络必须无缝处理有位姿和无位姿图像。我们的 DA3 主干网络满足这两个要求。特别是，<u>当有位姿时，我们通过 </u>_<u>[87]</u>_<u> 进行缩放并反投影预测的深度和相机空间中的 3DGS 到世界空间以对齐它们</u>。<u>当没有位姿时，我们直接使用预测的位姿进行反投影到世界空间</u>。

为减少准确表面几何与渲染质量之间的权衡 _[29]_，我们在 GS‐DPT 头中预测额外的深度偏移。为了提高在野外的鲁棒性，我们将每个 3D 高斯颜色替换为球面调和系数，以通过建模视图依赖的表面来减少与几何的冲突。

> _[87]_ Least-squares estimation of transformation parameters between two point patterns（IEEE Trans. Pattern Anal. Mach. Intell., 2002）：<u>提出了一种基于最小二乘的算法，用于估计两个点集之间的刚性变换参数（旋转、平移和缩放）</u>，在点云配准和姿态估计中广泛应用。
>
> _[29]_ Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering（CVPR 2024）：<u>提出了基于表面对齐高斯点的高效 3D 网格重建方法，通过预测额外深度偏移改善几何与渲染质量之间的权衡</u>，同时引入球面调和系数建模视图依赖的表面特性，提高野外场景的渲染鲁棒性。
>

#### Enhanced training strategies（增强的训练策略）
<u>为避免训练不稳定，我们在训练时从预训练权重初始化 DA3 主干网络，并冻结它，仅调整 GS‐DPT 头部</u>。<u>为了提高在野外的性能，我们使用不同图像分辨率和不同数量的上下文视图进行训练</u>。具体来说，高分辨率输入与较少的上下文视图配对，低分辨率输入则与更多的视图配对，这在稳定训练的同时支持了多样化的评估场景。

### <font style="color:rgb(0, 0, 0);">Implementation Details（实现细节）</font>
<font style="color:rgba(0, 0, 0, 0.85);">在训练 NVS 模型时，我们利用了大规模的 DL3DV 数据集 </font>_<font style="color:rgba(0, 0, 0, 0.85);">[53]</font>_<font style="color:rgba(0, 0, 0, 0.85);">，该数据集提供了多样化的真实世界场景，且包含由 COLMAP 估计得到的相机位姿。</font><u><font style="color:rgba(0, 0, 0, 0.85);">我们从 DL3DV 中选取了 10,015 个场景，用于训练前馈 3DGS 模型</font></u><font style="color:rgba(0, 0, 0, 0.85);">。</font><u><font style="color:rgba(0, 0, 0, 0.85);">为确保公平评估，</font></u><font style="color:rgba(0, 0, 0, 0.85);">我们严格保持训练集与测试集的互斥性：</font><u><font style="color:rgba(0, 0, 0, 0.85);">用于基准测试的 140 个 DL3DV 场景与训练集完全分离</font></u><font style="color:rgba(0, 0, 0, 0.85);">，避免了任何数据泄露。</font>

> _[53]_ DL3DV-10K: A large-scale scene dataset for deep learning-based 3D vision（CVPR 2024）：<u>构建了大规模、多样化的真实世界场景数据集 DL3DV-10K</u>，提供由 COLMAP 估计的相机位姿信息，支持前馈式 3D 表示学习模型的训练与评估，并通过严格的训练/测试集划分确保基准测试的公平性与无数据泄露。  
>

## Visual Geometry Benchmark
我们进一步引入了一个视觉几何基准测试，用于评估几何预测模型。它直接评估 位姿精度、通过重建的深度精度 和 视觉渲染质量。

### Benchmark Pipeline
#### Pose estimation（位姿估计）
<u>对于每个场景，我们选取所有可用图像；若总数超过限制，则使用固定随机种子随机采样 100 张图像</u>。将选中的图像送入前馈模型，生成一致的位姿与深度估计，随后计算位姿精度。

#### Geometry estimation（几何估计）
<u>对同一组图像，利用预测位姿与预测深度进行重建。为将重建点云对齐到真值</u>，我们使用 evo _[87]_ 将预测位姿对齐到真值位姿，得到一个将重建结果映射到真值坐标系的变换。为提高鲁棒性，我们采用基于 RANSAC 的对齐流程：在随机采样的位姿子集上反复运行 evo，并通过统计内点数量评估每个候选变换，其中内点定义为平移误差小于整体位姿偏差中位数的位姿。最终选择内点最多的变换，并将其用于 TSDF 融合，将已对齐的预测点云与预测深度图融合。最后，通过与真值点云对比，使用 1.7.2 节所述指标评估重建质量。

> _[87]_ Least-squares estimation of transformation parameters between two point patterns（IEEE Trans. Pattern Anal. Mach. Intell., 2002）：提出基于最小二乘的算法，用于估计两个点集之间的刚性变换参数（旋转、平移及缩放），广泛应用于点云配准和位姿对齐，是后续基于 RANSAC 的鲁棒对齐流程的理论基础。
>

#### Visual rendering（视觉渲染）
对于每个测试场景，所有基准测试数据集中的图像数量通常在 300 到 400 之间。我们从每 8 张图像中采样一张作为目标新颖视角进行评估。从剩余的视角中，我们使用每个数据集提供的 COLMAP 相机姿态，并根据相机平移和旋转距离进行最远点采样，选择 12 张图像作为输入上下文视角。对于 DL3DV，我们使用官方基准集进行测试。对于 Tanks and Temples，所有训练数据场景均被包含，除了法院。对于 MegaDepth，我们选择编号从 5000 到 5018 的场景，因为这些场景最适合 NVS。

### Metrics
#### Pose metrics（位姿指标）
<font style="color:rgba(0, 0, 0, 0.85);">在评估位姿估计时，我们遵循文献 </font>_<font style="color:rgba(0, 0, 0, 0.85);">[89, 91]</font>_<font style="color:rgba(0, 0, 0, 0.85);"> 中提出的评估协议，并使用 AUC 报告结果。</font><u><font style="color:rgba(0, 0, 0, 0.85);">该指标由两个部分组成：相对旋转精度（RRA）和相对平移精度（RTA）</font></u><font style="color:rgba(0, 0, 0, 0.85);">。RRA 和 RTA 分别量化了两幅图像之间旋转和平移的角度偏差。将每个误差与一组阈值进行比较，得到精度值。</font><u><font style="color:rgba(0, 0, 0, 0.85);">AUC 是精度 - 阈值曲线的积分</font></u><font style="color:rgba(0, 0, 0, 0.85);">，其中该曲线由每个阈值下 RRA 和 RTA 中的较小值确定。为了展示不同容忍度下的性能，</font><u><font style="color:rgba(0, 0, 0, 0.85);">我们主要报告阈值为 3 和 30 时的结果</font></u><font style="color:rgba(0, 0, 0, 0.85);">。</font>

> _[89]_ Posediffusion: Solving pose estimation via diffusion-aided bundle adjustment（CVPR 2023）：提出结合扩散模型与捆绑调整的方法用于位姿估计，通过扩散辅助优化提升位姿预测精度和鲁棒性，为复杂场景下的精确相机定位提供新思路。
>
> _[91]_ VGGT: Visual Geometry Grounded Transformer（CVPR 2025）：提出将视觉几何信息融入 Transformer 架构的方法，实现几何约束下的图像特征学习，提升了图像间相对位姿估计和几何感知任务的性能与泛化能力。
>

#### Reconstrution metrics（重建指标）
<font style="color:rgb(31, 35, 41);">令</font>![image](https://cdn.nlark.com/yuque/__latex/742feea1e00938322008014d1e5b27d2.svg)<font style="color:rgb(31, 35, 41);">表示真实点集，</font>![image](https://cdn.nlark.com/yuque/__latex/36f76032a83c9a35ab17dc424bde280c.svg)<font style="color:rgb(31, 35, 41);">表示待评估的重建点集。我们按照文献 </font>_<font style="color:rgb(31, 35, 41);">[1]</font>_<font style="color:rgb(31, 35, 41);"> 的方法，用</font>![image](https://cdn.nlark.com/yuque/__latex/8f81bd1cf5263c94fc743a06398ce753.svg)<font style="color:rgb(31, 35, 41);">衡量精度，用</font>![image](https://cdn.nlark.com/yuque/__latex/e1538dccb0f962837d3d32e9d786d916.svg)<font style="color:rgb(31, 35, 41);">衡量完整性，倒角距离（CD）定义为这两项的平均值。基于这些距离，我们针对距离阈值</font>![image](https://cdn.nlark.com/yuque/__latex/56c1b0cb7a48ccf9520b0adb3c8cb2e8.svg)<font style="color:rgb(31, 35, 41);">定义重建</font>![image](https://cdn.nlark.com/yuque/__latex/36f76032a83c9a35ab17dc424bde280c.svg)<font style="color:rgb(31, 35, 41);">的精确率和召回率： </font>

+ <font style="color:rgb(31, 35, 41);">精确率：</font>![image](https://cdn.nlark.com/yuque/__latex/b3c84957dec82aea1c84a5eba4c6c31b.svg)<font style="color:rgb(31, 35, 41);"> </font>
+ <font style="color:rgb(31, 35, 41);">召回率：</font>![image](https://cdn.nlark.com/yuque/__latex/c52406d68e4a68786933323c5a69796f.svg)<font style="color:rgb(31, 35, 41);"> </font>

<font style="color:rgb(31, 35, 41);">其中</font>![image](https://cdn.nlark.com/yuque/__latex/1dd723ec8dcb46a11ed3e0d5adb0118f.svg)<font style="color:rgb(31, 35, 41);">表示指示函数。为了同时体现这两个指标，我们报告 F1 分数，其计算公式为： </font>![image](https://cdn.nlark.com/yuque/__latex/74f3feb5b43c2e781609091e2d310c6a.svg)

> _[1]_ Large-scale data for multiple-view stereopsis（Int. J. Comput. Vis., 2016）：构建了大规模多视图立体匹配数据集，为评估和训练多视图立体重建算法提供了标准化基准；<u>提出了基于点云距离的精度、完整性和倒角距离（Chamfer Distance）评价指标，并引入精确率、召回率及 F1 分数用于综合衡量重建质量</u>。  
>

### Datasets
我们的基准 基于五个数据集构建：HiRoom [129]、ETH3D [72]、DTU [1]、7Scenes [74] 和 ScanNet++ [117]。它们涵盖了从以物体为中心的采集到复杂室内外环境等多样化场景，且在先前研究中被广泛采用。以下是数据集准备过程的详细信息： 

**HiRoom** 是由专业艺术家创建的 Blender <u>渲染合成数据集，包含 30 个室内生活场景</u>。计算 F1 重建指标时，我们使用的阈值![image](https://cdn.nlark.com/yuque/__latex/56c1b0cb7a48ccf9520b0adb3c8cb2e8.svg)为 0.005m；TSDF 融合的体素大小参数设为 0.007m。

**ETH3D **提供了<u>带激光传感器真实深度的高分辨率室内外图像</u>。我们通过 TSDF 融合聚合真实深度图以生成真实 3D 形状，选取了 11 个场景用于基准测试：courtyard、electro、kicker、pipes、relief、delivery area、facade、office、playground、relief 2、terrains。评估中使用所有帧，计算 F1 重建指标的阈值![image](https://cdn.nlark.com/yuque/__latex/56c1b0cb7a48ccf9520b0adb3c8cb2e8.svg)为 0.25m；TSDF 融合的体素大小参数设为 0.039m。

**DTU **是<u>包含 124 个不同物体的室内数据集，每个场景从 49 个视角记录</u>，提供了在受控条件下采集的真实点云。我们在 DTU 数据集的 22 个评估场景上评估模型，采用 RBMG 2.0 [126] 去除无意义的背景像素，并使用默认的 TSDF 融合策略[124]。所有帧均用于评估。

**7Scenes **是<u>具有挑战性的真实世界数据集，包含低分辨率且带有严重运动模糊的室内场景图像</u>。我们通过TSDF 融合 RGB-D 图像来生成真实 3D 形状；为方便评估，将每个场景的帧数下采样至 11。计算 F1 重建指标的阈值![image](https://cdn.nlark.com/yuque/__latex/56c1b0cb7a48ccf9520b0adb3c8cb2e8.svg)为0.05m；TSDF 融合的体素大小参数设为 0.007m。

**ScanNet++**是<u>大规模室内数据集，提供高分辨率图像、iPhone LiDAR 生成的深度图，以及激光扫描重建得到的高分辨率深度图</u>。我们选取 20 个场景用于基准测试：由于 iPhone LiDAR 缺乏无效深度指标，默认将激光扫描重建中采样得到的深度图作为真实深度。通过 TSDF 融合聚合真实深度图以生成真实 3D 形状；为方便评估，将每个场景的帧数下采样至 5。计算 F1 重建指标的阈值![image](https://cdn.nlark.com/yuque/__latex/56c1b0cb7a48ccf9520b0adb3c8cb2e8.svg)为 0.05m；TSDF 融合的体素大小参数设为 0.02m。

#### **Visual rendering quality（视觉渲染质量）**
我们在多样化的大规模场景上评估视觉渲染质量，基于三个数据集构建了新的神经视图合成（NVS）基准：包含 140 个场景的 DL3DV _[53]_、包含 6 个场景的 Tanks and Temples _[45]_，以及包含 19 个场景的 MegaDepth _[51]_，每个场景约有 300 个采样帧。真实相机姿态由 COLMAP 估计，直接用于确保不同模型间的准确公平比较。我们<u>报告 PSNR、SSIM 和 LPIPS 指标</u>，以渲染新视角并给定相机姿态。

> _[53] _DL3DV-10K: A large-scale scene dataset for deep learning-based 3D vision（CVPR 2024）：<u>构建了大规模、多样化的真实世界场景数据集 DL3DV-10K</u>，提供由 COLMAP 估计的相机位姿，用于训练和评估前馈式 3D 表示学习模型，并通过严格划分训练集与测试集确保公平比较。
>
> _[45]_ Tanks and Temples: Benchmarking large-scale scene reconstruction（ACM Trans. Graph., 2017）：提出大型场景重建基准 Tanks and Temples，包含真实场景的多视图数据与高质量重建参考，为评价多视图几何和重建算法提供标准化评价平台。
>
> _[51]_ MegaDepth: Learning single-view depth prediction from internet photos（CVPR 2018）：构建 MegaDepth 数据集，<u>通过从互联网照片自动生成深度图，用于单视图深度预测任务</u>；提出结合大规模网络训练和多视图几何约束的方法，显著提升单图像深度估计的准确性。
>

## Experiments
### Comparison with State of the Art
#### Baselines
VGGT _[91]_ 是一个端到端的 Transformer，能够从一个或多个视图中联合预测相机参数、深度和 3D 点。Pi3 _[99]_ 进一步采用置换等变设计，从无序图像中恢复仿射不变相机和尺度不变点图。 MapAnything _[43]_ 提供了一个前馈框架，也可以将相机姿态作为输入用于密集几何预测。Fast3R _[111]_ 将点图回归扩展到单次前向传递中处理数百甚至数千张图像。最后，DUst3R _[97]_ 通过回归点图并进行全局对齐来处理未校准的图像对。<u>我们的方法与 VGGT 类似，但采用了新的架构和不同的相机表示，并且与 Pi3 相互独立</u>。

> _[91] _VGGT: Visual Geometry Grounded Transformer（CVPR 2025）：提出端到端 Transformer 框架，可从一个或多个视图联合预测相机参数、深度与 3D 点，首次在大型场景中实现统一的几何感知与多视图推理。
>
> _[99] _π3: Scalable Permutation-Equivariant Visual Geometry Learning（arXiv 2025）：提出置换等变的几何学习架构，从无序图像中恢复仿射不变的相机与尺度不变点图，显著提升多视图几何在无序输入上的可扩展性与一致性。
>
> _[43] _MapAnything: Universal Feed-forward Metric 3D Reconstruction（arXiv 2025）：提出通用前馈式 3D 重建框架，可接受相机姿态作为输入并输出密集几何，避免传统 BA 优化，支持快速、可扩展的整场景重建。
>
> _[111]_ Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass（CVPR 2025）：将点图回归扩展到可同时处理上千张图像的前向传递，大幅提升大规模场景的重建速度，实现高效的端到端多视图结构推理。
>

#### Pose estimation
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765090515182-d8173859-7488-409e-9fbb-2e6902f23007.png)

**表 5. 与SOTA方法在位姿精度上的比较：**我们报告了AUC3 ↑ 和AUC30 ↑ 指标。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765090593802-ef98dc92-517e-4c03-bd14-c0e7818c6505.png)

**图 5. 位姿估计质量比较：**两个视频的相机轨迹如图所示。真实轨迹是通过在动态物体被遮挡的图像上使用 COLMAP 得出的。

如上所示，在 _表 2_ 和 _图 5_，<u>与五个基线相比，我们的 DA3‐Gaint 模型在几乎所有指标上均取得最佳性能</u>，唯一例外是在 DTU 数据集上的 Auc30。值得注意的是，在 Auc3 上我们的模型在所有竞争方法上至少实现了 8% 相对改进，而在 ScanNet++ 上，它在第二好的模型上实现了 33% 相对提升。

#### Geometry estimation（几何估计）
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765090836348-ad97d787-88ed-41d6-b3c0-a0025f5773a8.png)

**表 3**.**重建精度上的与SOTA方法的比较：**对于所有数据集（除 DTU 外），我们报告 F1 分数（F1 ↑）。对于 DTU，我们报告 倒角距离（CD ↓，单位:毫米）。w/o p. 和 w/ p. 表示无姿态和有姿态，表示重建时是否提供了真实相机姿态。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765090964630-22a7e969-1503-48bc-be52-451f1fa680d5.png)

**图 6.点云质量对比**：我们的模型生成的点云在几何上更加规整，并且比其他方法生成的点云噪声显著减少。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765091010097-dcc42b53-78f8-445a-a8de-0ff0d9d80f12.png)

**图 7. 深度图质量对比：**与其他方法相比，我们的深度图在各种场景中展现出更精细的结构细节和更高的语义正确性。

如图 _表 3_ 所示，<u>我们的 DA3‐Gaint 在几乎所有场景中都建立了新的 SOTA，优于所有竞争对手，在所有五种</u>**<u>无姿态设置</u>**<u>中均表现出色</u>。平均而言，DA3‐Gaint 实现了相对提升，优于 VGGT 25.1% 和优于 Pi3 21.5%。_<u>图 7</u>_<u> 和</u>_<u>图 6</u>_<u> 可视化了我们的预测深度和恢复的点云。结果不仅干净、准确、完整，还保留了细粒度几何细节，清楚地展示了其优于其他方法的优越性</u>。甚至更值得注意的是，我们的 DA3‐Large（0.30B 参数）在效率方面表现出色。尽管它 3× 更小，但在十个设置中有五个超越了之前的 SOTA VGGT（1.19B 参数），特别是在 ETH3D 上表现尤为突出。

<u>当相机姿态可用时，我们的方法和 MapAnything 都可以利用它们来获得改进的结果，其他方法也受益于真实值姿态融合</u>。我们的模型在大多数数据集上显示出明显的提升，除了 7Scenes，其中有限视频设置已经饱和了性能，并减少了姿态条件化的收益。<u>值得注意的是，使用姿态条件化时，扩大模型规模所带来的性能提升比无姿态模型更小，这表明姿态估计比深度估计对模型规模的依赖更强，需要更大的模型才能充分实现改进</u>。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765091186396-35f7fd84-1cea-4a9d-a6d6-fd7d563f357b.png)

**表 4. 单目深度对比。 δ1 ↑**

<u>单目深度精度也反映了几何质量</u>。如 _表 4 _所示，在标准单目深度基准测试中，我们的模型优于 VGGT 和 Depth Anything 2。<u>作为参考，我们也包含了教师模型的结果</u>。

#### Visual rendering
为了公平地评估前馈新视角合成 (FF‐NVS)，我们与三个最近的 3DGS 模型进行比较 pixelSplat _[11]_, MVSplat _[13]_ 和 DepthSplat _[108]_——并进一步通过用 Fast3R、MV-DUSt3R _[85]_ 和 VGGT 替换我们的几何主干网络来测试其他框架。所有模型均在统一协议下使用 DL3DV‐10K 训练集进行训练，并在我们的基准测试上进行评估。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1765091397124-bed2ed8d-af52-4291-b0b0-32bea96833d0.png)

**表 5.与SOTA方法在NVS任务上的比较：**我们在 NVS 任务上报告与现有前馈 3DGS 模型及其他主干网络方法的对比。对于每个场景，我们使用 12 个输入上下文视角，并在超过 300 个视角中每隔 8 个视角采样的目标视角上进行测试。图像分辨率为 270 × 480。

如 _表 5_ 所示，<u>所有模型在 DL3DV 上的表现都显著优于其他数据集，这表明基于 3DGS 的 NVS 对由 DL3DV 标准化的轨迹和姿态分布更为敏感，而不是场景内容</u>。<u>比较这两个组别，基于几何模型的框架始终优于专用前馈模型，这表明简单主干网络加上 DPT 头即可超越复杂的任务特定设计</u>。这种优势来自于大规模预训练，它比依赖极线变换器、代价体或级联模块的方法 在泛化能力和可扩展性上更优。在这个组别中，NVS 性能与几何估计能力相关联，使得 DA3 成为最强的主干网络。展望未来，我们预计 FF‐NVS 可以通过利用预训练的几何主干网络的简单架构得到有效解决，并且 DA3 强大的空间理解能力将有助于其他 3D 视觉任务。

> _[11]_ PixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction（CVPR 2024）：提出 PixelSplat，通过从两视图估计相对姿态与深度直接生成 3D 高斯点，实现高效、可扩展的前馈重建；方法具有强泛化能力，可在稀视图条件下快速获得点云级几何。
>
> _[13]_ MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-view Images（ECCV 2024）：提出 MVSplat，将多视图几何（特征匹配、深度估计）与高斯渲染融合，在稀视图输入下通过轻量级网络生成高质量 3D 高斯场，提高速度与效率。
>
> _[108]_ DepthSplat: Connecting Gaussian Splatting and Depth（CVPR 2025）：提出 DepthSplat，将深度估计和 3DGS 渲染统一为可微框架；通过显式连接深度场与可渲染高斯，提供更一致的几何建模，并支持端到端训练的前馈式重建与渲染。
>
> _[85]_ MV-DUSt3R+: Single-stage Scene Reconstruction from Sparse Views in 2 Seconds（CVPR 2025）：提出 MV-DUSt3R+，将 DUSt3R 扩展至多视图场景，通过单阶段前馈框架与快速全局对齐，仅需数秒即可从稀视图重建完整可扩展 3D 场景。
>

## Conclusion and Discussion
Depth Anything 3 表明，一个通过教师‐学生监督在深度和光线目标上训练的普通 Transformer 模型，可以在无需复杂架构的情况下统一任意视角几何。尺度感知深度、每个像素的光线以及自适应跨视角注意力使模型在继承强大的预训练特征的同时保持轻量和易于扩展。在提出的视觉几何基准测试上，该方法设定了新的姿态和重建记录，无论是大型还是紧凑型变体都超越了以往的模型，而相同主干网络则支持高效的前馈新视角合成模型。  

我们认为 Depth Anything 3 是迈向多功能 3D 基础模型的一步。未来的工作可以扩展其推理能力以适用于动态场景，整合语言和交互线索，并探索更大规模的预训练，以在几何理解和可操作世界模型之间建立闭环。 我们希望本研究提供的模型和数据集发布、基准测试以及简单的建模原则能够推动更广泛关于通用 3D 感知 的研究。







