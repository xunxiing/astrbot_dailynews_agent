# NewBieAi-lab发布NewBie-image-Exp0.1模型【AI 早报 2025-12-08】

- 作者：Juya
- 原文链接：https://mp.weixin.qq.com/s/K1FGe-TiAk52B8lD6Kg8Mw

---

## 正文

AI 早报 2025-12-08
概览
NewBieAi-lab发布NewBie-image-Exp0.1模型 #1
ServiceNow发布Apriel-1.6-15B-Thinker模型 #2
LLM360发布70B参数K2-V2-Instruct模型 #3
ImageCritic发布：参考引导图像后编辑框架 #4
Google DeepMind Gemini 3 Flash亮相LM Arena #5
Grok深度集成Tesla并预告4.20版本 #6
Stack Overflow发布AI Assist工具 #7
Google扩大TPU供应与数据中心部署 #8
d-Matrix 完成 C 轮融资 2.75 亿美元 #9
NewBieAi-lab发布NewBie-image-Exp0.1模型 #1

NewBieAi-lab发布了专为高质量动漫图像设计的开源DiT模型NewBie-image-Exp0.1，拥有3.5B参数。

NewBieAi-lab推出了NewBie-image-Exp0.1，这是一个拥有3.5B参数的开源DiT基础模型，专为精确、快速和高质量的动漫图像生成（ACG-native）设计。该模型是NewBie文本到图像生成框架的首个实验性版本，其架构基于对Lumina架构的研究，并以Next-DiT作为基础设计了新的NewBie架构。

NewBie-image-Exp0.1通过在包含超过10M张带有XML注释的高质量动漫数据语料库上预训练，使其能够生成细节丰富、视觉效果显著的动漫风格图像，并具备处理复杂多角色场景的能力。

该模型的模型权重和参数受Newbie Non-Commercial Community License (Newbie-NC-1.0) 许可约束，仅限非商业用途，且衍生品必须在相同许可下共享。而训练和推理脚本等相关源代码则采用Apache License 2.0。

https://www.modelscope.cn/models/NewBieAi-lab/NewBie-image-Exp0.1

ServiceNow发布Apriel-1.6-15B-Thinker模型 #2

ServiceNow发布了Apriel-1.6-15B-Thinker模型，在保持高性能的同时，将推理Token使用量减少了30%以上。

ServiceNow发布了Apriel-1.6-15B-Thinker模型，这是其Apriel SLM系列中更新的多模态推理模型。该模型在提升文本和图像推理能力的同时，实现了与高达其10倍大小的模型相媲美的竞争力性能。模型参数量为15B，具有高内存效率，可在单个GPU上运行。

相较于前身Apriel-1.5-15B-Thinker，新模型在改进或维持任务性能的同时，将推理Token使用量减少了 30% 以上，在实现前沿性能的同时优化了推理Token效率。

https://huggingface.co/ServiceNow-AI/Apriel-1.6-15b-Thinker

LLM360发布70B参数K2-V2-Instruct模型 #3

LLM360发布了其迄今为止最强大的70B参数开源模型K2-V2-Instruct，并公开了所有相关资源。

K2-V2-Instruct是LLM360发布的最新70B参数密集型Transformer架构模型，由Mohamed bin Zayed University of Artificial Intelligence（MBZUAI）Institute of Foundation Models（IFM）创建。该模型被定位为LLM360家族中迄今为止功能最齐全、最强大的完全开放模型之一。

LLM360作为一个开放研究实验室，旨在通过开源大型模型研究和开发，实现社区拥有的AGI，并为此公开了所有模型检查点、代码和数据集。

https://huggingface.co/LLM360/K2-V2-Instruct
https://github.com/llm360/k2v2_train

ImageCritic发布：参考引导图像后编辑框架 #4

ImageCritic是一个参考引导的图像后编辑框架，专门用于修正AI生成图像中的细粒度细节不一致问题。

ImageCritic是一个专门用于解决AI生成图像中细粒度细节不一致性问题的参考引导后编辑方法。该方法旨在解决现有定制化生成任务在生成连贯细节方面的局限性。

该框架的核心在于其技术组成：基于对模型注意力机制和内在表征的深入检查，研究人员设计了“attention alignment loss”和“detail encoder”。通过这些机制，ImageCritic能够精确地修正乱码文本、扭曲的Logo以及其他细粒度的不准确性，有效解决各种定制化生成场景中的细节相关问题，相比现有方法提供了显著改进，提升了图像质量和连贯性。

ImageCritic遵循知识共享署名-非商业性使用 4.0 国际许可（CC BY-NC 4.0），仅限非商业用途，任何商业用途需事先获得正式许可。

https://huggingface.co/ziheng1234/ImageCritic
https://ouyangziheng.github.io/ImageCritic-Page/

Google DeepMind Gemini 3 Flash亮相LM Arena #5

代号为skyhawk和seahawk的模型，被认为是Google DeepMind的Gemini 3 Flash，已出现在LM Arena上。

LM Arena 上线了代号为 skyhawk 和 seahawk 的模型。有观点认为，这是 Google DeepMind 研发的 Gemini 3 Flash 型，其出现表明 Google 正准备进行正式发布。

https://x.com/AILeaksAndNews/status/1997795213650415875

Grok深度集成Tesla并预告4.20版本 #6

xAI的Grok模型已深度集成至Tesla车辆，新增导航指令功能，并预告了Grok 4.20版本将在数周内推出。

xAI的Grok语言模型正在实现更深度的集成和功能扩展，主要体现在与Tesla车辆的整合，并宣布了下一代模型Grok 4.20的计划。

Grok已通过Tesla的2025 Holiday Release新版本推送，实现了更深层次的整合。该更新新增了“Grok with Navigation Commands (Beta)”功能，使Grok能够作为智能驾驶助手直接接受导航指令。用户可以口头指示Grok添加或编辑目的地，并由Grok引导行程。此外，用户在车辆内可以设置Grok的个性化特征。

针对未来的技术迭代，根据Elon Musk的说法，Grok 4.20版本预计将在大约三到四周内推出。

https://x.com/elonmusk/status/1997613405033967867
https://x.com/techdevnotes/status/1997642382028714415

Stack Overflow发布AI Assist工具 #7

Stack Overflow推出AI Assist工具，结合生成式AI与社区验证的知识，旨在提供更准确、可信的答案。

Stack Overflow正式推出AI Assist工具，旨在适应开发者在AI时代获取知识方式的转变，并降低用户使用社区的门槛。AI Assist是一种快速高效的学习工具，它将生成式AI的能力与Stack Overflow社区积累的十七年专家知识相结合，通过人类验证的内容来提供答案。

该工具的核心理念是“准确优先，逻辑为纲”，在设计上将引用、归属和人为验证的答案视为不可妥协的要素，以解决用户对AI工具准确性的信任问题，尽管AI的使用率持续上升，但用户对AI信任度有所下降。

https://stackoverflow.blog/2025/12/02/introducing-stack-overflow-ai-assist-a-tool-for-the-modern-developer
http://stackoverflow.com/ai-assist

Google扩大TPU供应与数据中心部署 #8

Google正通过与Fluidstack合作扩大TPU数据中心部署，并计划在2027年前生产超过500万颗TPU。

Google正通过扩大基础设施合作与设定高生产目标，积极强化其TPU（Tensor Processing Unit）供应与生态系统。

在基础设施部署方面，Google已同意为Fluidstack公司提供租赁担保，以支持Fluidstack对三个正在开发中的数据中心的使用。作为此项协议的一部分，Fluidstack同意至少在一个位于纽约的数据中心内托管Google的TPUs。

在TPU生产方面，Google规划在2027年之前生产超过500万颗TPUs，以大幅增加供应。

https://x.com/theinformation/status/1997708945067032863
https://thein.fo/48TE03u
https://x.com/theinformation/status/1997754485955789093
https://thein.fo/3Ka5j05

d-Matrix 完成 C 轮融资 2.75 亿美元 #9

专注于AI推理计算的d-Matrix完成了2.75亿美元的C轮融资，公司估值达到20亿美元。

d-Matrix 是一家专注于数据中心生成式 AI 推理计算的先行者，已完成 2.75亿美元 的 C轮 融资，使公司总融资额达到 4.5亿美元，估值达到 20亿美元。本轮超额认购的资金将用于推进公司的产品路线图、加速全球扩张，并支持针对超大规模、企业和主权客户的多次大规模部署，以满足市场对更快、更高效数据中心推理日益增长的需求。

https://www.d-matrix.ai/announcements/d-matrix-raises-275-million-to-power-the-age-of-ai-inference/


提示：内容由AI辅助创作，可能存在幻觉和错误。

作者橘鸦Juya，视频版在同名哔哩哔哩。欢迎点赞、关注、分享。


## 图片

![image_0](images\img_0.png)
![image_1](images\img_1.jpeg)
![image_2](images\img_2.png)
![image_3](images\img_3.png)
![image_4](images\img_4.png)
![image_5](images\img_5.png)
![image_6](images\img_6.png)
![image_7](images\img_7.png)
![image_8](images\img_8.png)
