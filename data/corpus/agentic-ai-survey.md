# Generative to Agentic AI: Survey, Conceptualization, and Challenges

Johannes Schneider

University of Liechtenstein, Vaduz, Liechtenstein.

Contributing authors: johannes.schneider@uni.li;

## Abstract

Agentic Artificial Intelligence (AI) builds upon Generative AI (GenAI). It constitutes the next major step in the evolution of AI with much stronger reasoning and interaction capabilities that enable more autonomous behavior to tackle complex tasks. Since the initial release of ChatGPT (3.5), Generative AI has seen widespread adoption, giving users firsthand experience. However, the distinction between Agentic AI and GenAI remains less well understood. To address this gap, our survey is structured in two parts. In the first part, we compare GenAI and Agentic AI using existing literature, discussing their key characteristics, how Agentic AI remedies limitations of GenAI, and the major steps in GenAI's evolution toward Agentic AI. This section is intended for a broad audience, including academics in both social sciences and engineering, as well as industry professionals. It provides the necessary insights to comprehend novel applications that are possible with Agentic AI but not with GenAI. In the second part, we deep dive into novel aspects of Agentic AI, including recent developments and practical concerns such as defining agents. Finally, we discuss several challenges that could serve as a future research agenda, while cautioning against risks that can emerge when exceeding human intelligence.

Keywords: Agentic AI, Generative AI, Conceptualization, Survey

## Contents

- Introduction (p. 3)
- Generative AI to Agentic AI (p. 5)
  - Defining GenAI and Agentic AI (p. 5)
  - Capabilities Compared (p. 8)
  - Why Agentic AI? (p. 9)
  - Evolution of GenAI to Agentic AI (p. 11)
- Agentic AI (p. 13)
  - Reasoning (p. 13)
    - Decomposition (p. 14)
    - Reflection (p. 15)
    - Planning and Search (p. 17)
    - Learning to Reason (p. 18)
  - Memory (p. 20)
    - Memory Types and Structures (p. 20)
    - Retrieval Augmentation (p. 21)
  - Tools (p. 22)
    - Tool Creation (p. 22)
    - Tool Selection (p. 23)
    - Tool Use (p. 23)
  - Interacting (p. 23)
    - Reinforcement Learning (p. 23)
    - Interacting in (Virtual) Worlds (p. 24)
    - Interacting with Humans (p. 27)
    - Interacting with Other Agents (p. 28)
  - Specifying and Evaluating Agentic AI (p. 28)
    - Specifying an Agent (p. 28)
    - Specifying Multi-Agents (p. 29)
    - Evaluating Agentic AI (p. 31)
- Challenges Toward AGI (p. 32)
  - Challenges (p. 32)
  - Agentic AI to AGI? (p. 33)
- Methodology and Related Work (p. 33)
  - Methodology and Scope (p. 33)
  - Related Surveys (p. 34)
- Conclusion (p. 35)

## 1 Introduction

Agentic AI constitutes a paradigm shift in artificial intelligence, enabling systems to act independently, pursue broad objectives rather than isolated decisions, and carry out complex tasks that require reasoning elements such as planning and reflection. While it builds on earlier seeds of shallow reasoning and interaction already observed in early GenAI systems, Agentic AI extends and structures these capabilities profoundly. Agents (i) interact with environments and tools, and (ii) perform deep reasoning, which manifests as multi-step, problem-dependent computation with planning and reflection. For example, reasoning models such as ChatGPT o1 can spend minutes processing self-generated prompts as part of a search and planning process, whereas earlier GenAI models like ChatGPT 3.5 (released in 2022) typically provided instantaneous responses, as shown in Figure 1.

**Fig. 1** Reasoning models perform extensive problem-dependent computations, commonly employing problem analysis, planning and reflection, while non-reasoning is shown by immediate responses without intermediate steps.

Beyond limited reasoning capabilities, tasks handled by GenAI are typically structured to be solved directly using the available information--that is, by generating an output from the input without engaging with the environment or external tools. In contrast, Agentic AI systems integrate elements of reinforcement learning: Agents interact with environments using tools through a sequence of actions, receiving feedback that informs and guides future actions via instant learning. This new wave of AI fosters novel opportunities and challenges, making it crucial to understand and conceptualize differences clearly. Defining an autonomous agent is more challenging and riskier, as agents have a greater direct impact: they interact with the

**Fig. 2** On the ARC challenge (Clark et al., 2018) Agentic AI with its reasoning models such as o1 and o3 performing dynamic, extensive computations involving planning and reflection dramatically outperform other models.

environment by taking actions and solve tasks with less-detailed instructions, leading to more diverse, unpredictable, and harder-to-control solutions. Moreover, capabilities such as autonomous memory use, flexible tool selection, and open-ended exploration amplify both the potential benefits and risks compared to GenAI. On the positive side, AI agents, like GenAI, can be used with plain, everyday language instead of complex programming languages, boosting ease of adoption. On the negative side, properly specifying an agent to account for various eventualities is more demanding than crafting a prompt for a narrower task. While some experts already warned against the threats of GenAI (Heath, 2023), many researchers deemed those threats as overstated (Nolan, 2024). This is not a surprise, as GenAI models struggled to solve tasks that were easy for laypeople. For instance, in the ARC challenge (Clark et al., 2018)--which consists of tasks considered simple for ordinary people--early GenAI models and even more recent ones perform poorly. However, reasoning models that are part of Agentic AI show striking improvements, as illustrated in Figure 2. In other areas, such as programming (El-Kishky et al., 2025) and generating coherent and accurate radiology reports, Agentic AI models with reasoning capabilities also outperform other models (Zhong et al., 2024). These outcomes also suggest that Agentic AI represents a major step toward artificial general intelligence (AGI), where AI is not confined to a narrow domain but can generalize to new situations. However, scaling compute at test-time alone (a key aspect of deep reasoning) may not guarantee progress toward AGI, and important challenges remain, including training data limitations and error accumulation. Thus, the debate on the risks and opportunities of AI should intensify again, requiring a renewed and cautious assessment. Understanding the differences between the two is important for businesses to choose the right technology to save time and reduce costs, and for society and policymakers to engage in a meaningful ethical and social debate surrounding the technology (Marr, 2025).

However, the situation becomes more intricate as Agentic AI is not a magic bullet for all problems. In particular, it does not offer dramatic improvements over GenAI in all areas. This makes it challenging to assess which problems benefit from clear-cut gains that enable novel applications. Like GenAI, Agentic AI can make mistakes on relatively simple problems (Zhong et al., 2024). Social intelligence is not

necessarily higher in models performing deep reasoning (Hou et al., 2024), although differences may emerge in tasks requiring the anticipation of behavioral strategies, such as those found in games. Only minor improvements have been observed on translation tasks (Chen et al., 2025). On the well-known MMLU benchmark--consisting of multiple-choice questions spanning factual and procedural knowledge from various domains--reasoning models show only a slight edge, as illustrated in Figure 4, and at the cost of significantly greater computation. Many of these questions are considered "hard" in the sense that they require PhD level knowledge in specific domains. Likely, there are a number of analogous questions to those in the benchmark in the model's training data - in the most extreme case even the questions and answers themselves. Thus, the questions may be easier to answer through association rather than genuine reasoning. Generally, fine-tuning GenAI models for specific domains may outperform Agentic AI models that lack domain-specific knowledge, while also incurring lower computational costs.

Contribution and Overview: This manuscript enables the reader to better understand the novel risks and opportunities of Agentic AI through various conceptualizations and contextualizations. As detailed in Figure 3, we first definitions contrasting GenAI and Agentic AI and compare their high-level capabilites. We also elaborate on limitations of GenAI and how Agentic AI overcomes (some of) them. While Agentic AI is often portrayed as an unforeseeable, sudden change causing instant disruption, we document its evolution from early GenAI systems, highlighting key milestones related to reasoning and interaction, and noting overlaps and differences with well-established AI paradigms such as reinforcement learning. Next, we provide a deep dive into Agentic AI. We discuss key aspects of Agentic AI, including reasoning (decomposition, multi-step reasoning, reflection, search and planning, memory, tools and interaction with the environment covering also a range of applications). We elaborate on how to specify AI agents, including both single-agent and multi-agent systems as well as how to evaluate Agentic AI systems. Finally, we describe the challenges of Agentic AI to further progress towards AGI, which constitute research opportuntities. We also caution on potential risk coming with AGI. Before concluding we elaborate on our research methodology and related surveys.

## 2 Generative AI to Agentic AI

We start with defining both GenAI and Agentic AI before elaborating on their key characteristics, discussing the need for Agentic AI and the evolution of GenAI to Agentic AI from a research perspective.

### 2.1 Defining GenAI and Agentic AI

While many academic works have some notion of Agentic AI, we found that different angles of it were most concisely described by leading AI companies: "Agentic AI uses sophisticated reasoning and iterative planning to autonomously solve complex, multi-step problems." (NVidia (Pounds, 2024)) "Agentic AI systems are AI systems that can pursue complex goals with limited direct supervision" (OpenAI (Shavit et al., 2023)).

**Fig. 3** Overview: The high-level discussion of GenAI to Agentic AI is followed by an in-depth treatment of Agentic AI, followed by challenges and outlook

We add perspectives emphasizing more the differences to GenAI. We start with a short definition focusing on high-level, non-technical aspects: "Generative AI generate and transform content based on specific user instructions." "Agentic AI can autonomously execute complex tasks in dynamic environments requiring adaptation, interaction and reasoning in a goal-oriented manner." This definition emphasizes differences in characteristics as shown in Table 1.

A more comprehensive definition with a more academic phrasing: Definition of Generative AI: "A system based on a foundation model that generates digital artifacts based on natural user instructions." Digital artifacts can be anything ranging from simple binary decisions to text, images, and images. Natural user instructions refers to instructions that are easy to understand for humans and do not require deep technical or specific knowledge of the AI system, e.g., human language, or visual depiction such as photos or sketches.

**Fig. 4** Agentic AI and GenAI performs similarly on MMLU (Hendrycks et al., 2021b) measuring a wide range of capabilities across disciplines(data from (Paperswithcode, 2025))

Definition of Agentic AI: "A system based on a foundation model that performs tasks potentially yielding artifacts based on natural user instructions, where the system is able to conduct and express complex reasoning including planning and reflection to solve tasks that require interaction with an environment or elaborate tool use." Agentic AI can go beyond creating digital artifacts, as it might, e.g., control robots to produce or alter physical objects. It might also execute tasks that do not yield any directly observable artifact for a user, e.g., smart grid management or computer internal system optimizations such as scheduling computational resources to minimize costs while obeying constraints. That is, while an agent takes action that impact our world, the outcome is not a tangible artifact such as a text or images. In both definitions we included the term foundation model, which is a large deep learning network (Schneider et al., 2024) though agents could be (and have been) built with other model types. However, many general and more detailed, technical aspects expressed in our and other works assume the usage of a foundation model as otherwise many claims are not substantiated.

When emphasizing the interconnection between prior AI developments, one might say that: "Agentic AI builds on Generative AI by combining foundation models enhanced with the capability for tool usage, memory access with reinforcement learning with its notion of agents and planning to enable interaction and reasoning." In contrast, GenAI only relies on foundation models with early models having only limited capabilities for tool usage and reasoning - This definition is aligned with Figure 5.

A foundation model is to GenAI what an agent is to Agentic AI. Agents themselves can be systems including a foundation model (see Figure 5). However, this is somewhat oversimplified, as the construct of agents in AI is rather old and goes beyond agents being a technical component (Russell and Norvig, 2021). Agentic AI is a subfield of AI, while an AI agent is the central object of study. Thus, Agentic AI is more comprehensive including procedures to training, evaluating, and defining agents, and coordinating multiple agents. It also covers non-technical aspects such as ethical,

**Fig. 5** From Generative AI to Agentic AI. Early GenAI such as GPT-3 (Brown et al., 2020) showed with minor effort basic reasoning and tool usage capabilities. These became much more profound over time, in particular, with Agentic AI, which includes elements from reinforcement learning such as interacting with the environment and other agents.

economic, social and philosophical debates. For brevity we shall often omit the term "systems" in combination with GenAI and Agentic AI.

| Aspect | Generative AI | Agentic AI |
|---|---|---|
| Reasoning | ✗immediate responses | ✓iterative planning and reflection |
| Interaction | ✗mostly only with user | ✓user, tools, real-world, other AI agents |
| Execution capability | ✗single-step tasks | ✓ Workflows, sequence of actions requiring diverse expertise |
| Adaptability | ✗ no self-improvement, bound to training data | ✓Collecting and leveraging experiences |
| Autonomy | ✗user-driven | ✓self-directed |

### 2.2 Capabilities Compared

In terms of their capability, Agentic AI differ from early GenAI in two foundational aspects: (i) Reasoning and (ii) interaction with an environment and tools. Other characteristics stated in the literature, such as adaptability, execution capability, and autonomy (shown in Table 1), depend heavily on reasoning and interaction. Regarding autonomy, GenAI can be viewed as assistive technology for simple tasks, requiring specific user instructions for each interaction, whereas Agentic AI demonstrates at least partial autonomy by executing multi-step tasks (see Table 2) (Agent.ai, 2025).

| Autonomy Level | Paradigm | Description |
|---|---|---|
| 0 - No Autonomy | Classical Machine Learning | Tackles narrow tasks it was explicitly trained for. |
| 1 - Assistive Autonomy | Generative AI | Handles simple tasks with direct instruction. |
| 2 - Partial Autonomy | Agentic AI: Agent-Oriented Workflow | Manages multi-step tasks with human oversight and intervention. |
| 3 - High Autonomy | Agentic AI: Goal-Oriented Collaboration | Achieves complex tasks with occasional guidance. |
| 4 - Full Autonomy | Agentic AI: Autonomous Decision-Making | Given goals, handles all aspects of tasks independently. |

Agentic AI allows users to define their own agents using less detailed descriptions that may include identity, professional role, constraints, tool-use capabilities, etc. Park et al. (2023); Hong et al. (2023); see Figure 15 for a concrete example. However, as of now, autonomy for both GenAI and Agentic AI may be severely constrained by legislation requiring human oversight (European Union, 2023). Also adaptability including autonomous self-improvement and instant learning from experiences over a long timespan, is difficult as evidenced by prominent failures of earlier systems (BBC News, 2016; CDOTrends, 2022) and remains limited for Agentic AI. While many works focus on "LLM agents," generalist agents possess multi-modal capabilities and can perform a wide range of tasks, including playing Atari games, controlling a real robot arm, and engaging in ordinary conversation (Reed et al., 2022).

### 2.3 Why Agentic AI?

Generative AI comes with a series of shortcomings some of which are shown in Table 1. They motivate further development. More specifically, Agentic AI promises to eliminate or at least reduce some GenAI limitations, as summarized in Table 3. One of the most pressing limitations of GenAI is its execution capability. Current state-of-the-art GenAI models struggle with moderately complex tasks that require multiple actions, such as simple browser interactions (Drouin et al., 2024; Zhou et al., 2023), e.g., ordering a product in a webshop. Although early GenAI demonstrated basic reasoning and tool usage capabilities, it remained highly limited--similar to early AI systems like ELIZA in the 1960s, which could chat with humans but in an unsophisticated manner.

This lack of capability can be partially compensated through sophisticated prompt engineering, such as providing detailed instructions on how to approach a problem. However, reasoning models require only high-level goals and can derive detailed solution steps on their own, thereby enhancing usability.

The paradigm of scaling training data, models, and compute led to the breakthrough of foundation models (Brown et al., 2020; Kaplan et al., 2020; Chung et al., 2024). Although this approach may continue to boost AI's capabilities, it is not necessarily optimal. First, there is a significant lack of training data. Data can be expensive to collect--or, in some cases, impossible to obtain. It is not feasible to anticipate all potential tasks and gather large amounts of domain-specific training data for each.

Early GenAI models had very limited memory. For example, ChatGPT 3.5 could only handle inputs of about 4000 words without access to external data sources. More modern foundation models part of Agentic AI can handle millions of input tokens and perform sophisticated retrieval of information to stay up-to-date, support reasoning, and reduce errors.

Early GenAI generally lacked the capability to learn instantly. GenAI's limited context window allowed it to consider only a few past user interactions, making it incapable of remembering context or feedback in long conversations. Modern models have much larger context windows and can dynamically retrieve relevant information from databases. Agentic AI systems can learn in a more trial-and-error fashion--e.g., by simulating potential outcomes (Yao et al., 2023) or incorporating real-time feedback from users or tools, such as compiler error messages for generated code.

However, even for tasks that GenAI handles well, Agentic AI's reasoning capabilities can reduce errors such as hallucinations--i.e., generated content that is unfaithful to the input (Maynez et al., 2020). These may either contradict the source content (intrinsic hallucinations) or be unverifiable (extrinsic hallucinations). Agents that perform self-verification and retrieve external knowledge tend to exhibit lower error rates (Lewis et al., 2020; Dziri et al., 2021).

Additionally, academic literature has extensively discussed other GenAI shortcomings, such as biases and interpretability, which remain unresolved despite significant efforts by both industry and academia (Schneider, 2024a; Zoe Kleinmann, 2024). Agentic AI improves transparency and interpretability by providing intermediate results allowing for easier verification and better understanding. Furthermore, Agentic AI's reasoning capabilities offer the potential to reduce biases.

Agentic AI enables dynamic and flexible dedication of resources during inference. For example, to improve performance for a specific task one might invest more computation as shown in Figure 2, where tripling the spending in monetary terms improved performance from 25 to 32%, a relative gain of over 20%. GenAI can only approximate this behavior coarsely by switching between models of different sizes, such as small and large variants. In contrast, an agent encapsulating a single model allows fine-grained control of computation based on intermediate outputs. When resources like computation or electricity are limited, reasoning time--and thus response quality--can be reduced to serve more requests. Computational resources are spent both during model training and later during inference, i.e., when handling user tasks.

Agentic AI enables novel configurations for improving cost-efficiency and amortization of AI products. Smaller, faster, and less costly models can achieve performance comparable to large models by incorporating reasoning. This makes it economically viable to develop specialized models that are infrequently used, as they require less amortization. Similar to automation in manufacturing, one can choose between high startup costs with low operating costs and limited flexibility (large models), or low startup costs with dynamic, though potentially higher, operating costs (small models with reasoning). Figure 6 illustrates the performance of a large model using a single generation (i.e., one prompt execution) versus a small model with reasoning that produces a variable number of generations to solve self-generated sub-tasks (Beeching et al., 2025).

**Fig. 6** A small model with 1 billion parameters (1B) can outperform a larger 8B model in accuracy as shown for the MATH-500 benchmark using more computation, i.e., generations meaning calls to the model. Different "reasoning" strategies strongly impact the effectiveness of generations. Figure based on (Beeching et al., 2025).

| Aspect | GenAI Limitation | Agentic AI Advantage |
|---|---|---|
| Execution capability | ✗Failing multi-step tasks. Limited to generation of digital content with limited tool usage. | ● Performs multiple steps to solve tasks using planning, interaction with an arbitrary environment and tools. |
| Usability | ✗Requiring (rather) detailed task execution instructions | ✓ Goals without detailed instructions are sufficient. |
| Training data | ✗ Relies on comprehensive taskspecific training data, which may be infeasible or expensive. | ●Leverages logic and external tools to operate with less data. |
| Memory | ✗small context-windows | ✓ Larger context memory learning abstractions with storage and retrieval from databases. |
| Instant learning | ✗Limited by small context window | ✓ Unlimited memory; trial and error learning through simulation and realworld interactions. |
| Errors | ✗ Errors such as hallucinations are common | ● Less errors due to step-by-step reasoning and validation. |
| Transparency & interpretability | ✗Hard to interpret. | ● Shows intermediate reasoning, making outcomes explainable, though still hard to interpret. |
| Dynamic and flexible dedication of resources during operation | ✗ Little control over computational resources per task | ✓ High control allowing to perform or less reasoning impacting solution quality and costs. |
| Cost-efficiency and amortization of AI products | ✗High upfront training costs for large models requiring frequent usage for amortization | ✓ Low up-front costs due to smaller models with reasoning, supporting economically viable deployment even for infrequent tasks. |

### 2.4 Evolution of GenAI to Agentic AI

From a historical perspective, GPT-2 (Radford et al., 2019) (2019), GPT-3 (Brown et al., 2020) (2020), and the public release of ChatGPT 3.5 (OpenAI, 2022) (2022)--which resulted from post-training GPT-3 for human alignment and instruction following (Ouyang et al., 2022)--marked major breakthroughs in text generation,

with substantial differences in performance and task coverage among these models. In this work, we consider these models as marking the dawn of modern Generative AI, as they enabled prompt-based, controlled text generation of significantly higher quality than before. These "foundation models" could address a wide variety of tasks without task-specific training (Schneider et al., 2024). Technical foundations such as transformers date back further in time (Schneider, 2024b). Additionally, models like Generative Adversarial Networks (GANs) (Goodfellow et al., 2014; Zhang et al., 2023) enabled high-quality image generation years earlier, but lacked controllability--especially through natural language instructions.

Evolution of reasoning: Early generative AI models (Radford et al., 2019; Brown et al., 2020; OpenAI, 2022) typically responded to requests instantly. That is, the input was passed through the model once, without generating intermediate outputs, directly yielding a response--e.g., the prompt "7 * 12 =" producing "84". However, with suitable prompting (Wei et al., 2022b; Kojima et al., 2022), even early models like GPT-3 (Brown et al., 2020) demonstrated basic shallow reasoning, producing longer outputs that detailed the steps taken to reach an answer. These reasoning capabilities became more advanced, as shown in benchmark results (Figure 2), due to training on reasoning trace data (OpenAI, 2025) and the application of specialized techniques like planning. Deeper reasoning in Agentic AI systems requires significantly more computation, involving exploration of multiple options (planning and search) that are evaluated and refined through reflection. Scaling computation at inference time allows dynamic control over solution quality, as illustrated in Figure 6.

Several early prompting patterns used in GenAI to enhance reasoning have been integrated into the reasoning processes of Agentic AI. To illustrate this, we highlight a few patterns from (White et al., 2023). The "alternative approaches pattern" asking for multiple diverse solutions is found in multiple planning and search methods, such as ToT and FoT. The "question refinement pattern" is inherently part of the reflection process in reasoning. The "cognitive verifier pattern" follows the idea of problem decomposition (Kojima et al., 2022), which is a core part of reasoning. The "recipe pattern" translates a goal into a sequence of steps, which is also done by agents.

Evolution of interaction, tools, and memory: Early GenAI was also capable of basic interactions with tools like calculators and external memory sources such as databases (Schick et al., 2023). Retrieval-augmented generation (RAG), which uses external data from vector databases, marked a key milestone (Lewis et al., 2020) and was quickly integrated into commercial systems like GPT-4. These capabilities have become more advanced in modern Agentic AI systems. Furthermore, context window sizes--functioning as short-term, input-dependent memory--have increased from a few thousand to millions of tokens, enabling key Agentic AI features such as instant learning from experience.

Early GenAI focused on tasks that could be solved with a single generated output and little or no interaction with the environment. In contrast, Agentic AI increasingly incorporates the established paradigm of reinforcement learning (RL), "in which an agent interacts with the world and periodically receives rewards that reflect how well it is doing" (Russell and Norvig, 2021). RL targets tasks that involve a sequence of

actions, each influencing the environment. An agent may continuously sense its environment and periodically receive feedback on its performance. Solving a task may require an agent to make multiple attempts, learning from both failures and successes. Agents also face challenges such as exploitation (using existing knowledge) and exploration (acquiring new knowledge) (Russell and Norvig, 2021). That is, exploration aims at getting more knowledge about poorly understood aspects of the environment through novel behaviors and learning from observations to derive better solutions. In contrast, exploitation uses the agent's existing knowledge to solve a task, typically yielding only incremental insights. During exploitation a person might take the shortest way to work as every day as it is the fastest known route. During exploration a person might take the subway for the first time. From narrow prompting to defining autonomous agents: Agents are customized versions of foundation models, potentially incorporating orchestration functionalities, e.g., for tool access. Agents are typically defined via textual descriptions that include roles or personas, workflows, and permitted tool usage (Park et al., 2023); see Figure 15. To solve a concrete task, an agentic system can simply be prompted as GenAI. The system then engages in a dynamic, stateful solution process, often generating a series of prompts, as seen in early Agentic AI platforms like AutoGPT (Richards, 2023). However, GenAI also adopted customization of foundation models through prompting. For example, defining roles and personas has gained attention as a key prompting pattern (Wang et al., 2023; White et al., 2023). This has become a standard prompting technique in GenAI. By incorporating such definitions into the system prompt (Lee et al., 2024), users can effectively create customized foundation models. This process is supported by commercial platforms such as OpenAI's GPT Store (OpenAI, 2023), which facilitates easy sharing of models with specific system prompts and potentially private data. Additionally, the level of abstraction differs when defining an agent versus specifying a prompt in a GenAI context. For example, defining workflows in Agentic AI is arguably more high-level as reasoning models are expected to fill in more detailed, missing steps compared to more low-level instructions in GenAI (as done in early chain-of-thought prompting (Wei et al., 2022b)). Moreover, reasoning is now a built-in feature of Agentic AI, whereas in GenAI it had to be elicited using specific prompting patterns (White et al., 2023).

## 3 Agentic AI

We elaborate on three essential areas that distinguish Agentic AI and GenAI: reasoning (problem decomposition, verification, search and planning), interaction (with the environment, tools, and memory) and specification of single and multi-agents systems.

### 3.1 Reasoning

Agentic AI performs reasoning. Reasoning can be broadly defined as "the action of thinking about something in a logical, sensible way" (Oxford Languages, n.d.). In the AI literature multiple types of reasoning are discussed such as inductive reasoning (generalizing from examples and experiences) and deductive (applicaiton of rules). Analogical prompting has also been used as a reasoning paradigm for LLMs (Yasunaga

**Fig. 7** Chain-of-thought(CoT) detailing steps to solve a task in the input elicit a CoT in the response thanks to in-context learning (Figure adjusted from (Wei et al., 2022b))

et al., 2023). However, the most prevalent form applied by agent is the creation and execution of a step-by-step solution process (Kojima et al., 2022; Wei et al., 2022b) at inference involving, potentially, problem analysis, planning, solving, reflection with validation and refinement. This contrasts with an immediate, intuitive memory-based response--analogous to Kahneman's concepts of fast and slow thinking. In fact, some approaches allow agents to dynamically choose between slow and fast thinking (Lin et al., 2023). Technically, a reasoning model does not immediately provide the task solution; instead, it first generates intermediate results for self-defined subproblems--e.g., for the prompt "7 * 12 =", it might decompose the task into "2 * 7 = 14, 7 * 10 = 70, 14 + 70 = 84" rather than directly responding with "84". Alternatively, it might first decompose the problem into "2 * 7 = a, 7 * 10 = b, a + b = result" and then solve each subtask to arrive at the final answer. Shallow and deep reasoning: Shallow reasoning appeared in early GenAI systems, typically triggered by prompting (Kojima et al., 2022; Wei et al., 2022b). However, evaluation of the reasoning capability of LLMs suggest that they rely on patterns and correlations found in training data rather than on reasoning abilities (Mondorf and Plank, 2024). Furthermore, (simple) CoT primarily helps on math and symbolic reasoning (Sprague et al., 2024). Deep reasoning is more extensive, better structured, and pursued automatically in modern Agentic AI systems. It often integrates algorithms--e.g., for planning (Yao et al., 2023)--and tool use, where problems are translated into code and solved via a code interpreter (Gao et al., 2023; Chen et al., 2022).

#### 3.1.1 Decomposition

Multi-step solutions: Problems can be solved through multi-step reasoning in various ways--either via a single network pass generating a long answer or through a more intricate process involving multiple agents with separate planning and reflection, as shown in Figure 1. The latter is the more prevalent paradigm in Agentic AI. Basic decomposition can be achieved through prompting, e.g., by providing a multi-step reasoning example as in Chain-of-Thought (CoT) (Wei et al., 2022b) (see Figure 7), asking the model to "Think step-by-step" (Kojima et al., 2022), or explicitly instructing it

to decompose and then solve each subtask, as in least-to-most decomposition (Zhou et al., 2022). Recent work also showed that decoding strategies can elicit reasoning, in particular, CoT generations have higher confidence (Wang and Zhou, 2024).

"Planning is defined as the task of finding a sequence of actions to accomplish a goal" (Russell and Norvig, 2021). "The computational process of planning is called search" (Russell and Norvig, 2021). For instance, tree search algorithms aim to identify a sequence of actions that leads to a goal. Early large language models typically did not perform explicit search before solving a task but relied on prompting to execute step-by-step reasoning (Wei et al., 2022b; Kojima et al., 2022). When designing reasoning demonstrations for prompts, as in in-context learning in the classic Chain-ofThought paper (Wei et al., 2022b), two key dimensions emerge: (i) content within each reasoning step (e.g., answer accuracy, use of reasoning keywords) and (ii) structure (e.g., reflection, validation, logical coherence). Recent work has shown that structure is much more important (Li et al., 2025). Moreover, the length of reasoning chains--such as decomposition into smaller problems--is more impactful than the difficulty of individual components (Shen et al., 2025). Though longer chains often perform better, verbosity is not helpful and concise chains-of-thought can be more effective (Xu et al., 2025).

Hierarchical decomposition: While CoT resembles sequential planning, the concept of hierarchical planning has also been explored (Ajay et al., 2023; Yang et al., 2024; Wu et al., 2024). In (Ajay et al., 2023), high-level steps are proposed and then refined into more concrete geometric and control-level actions. Different levels of abstraction use different models--for instance, a large language model (LLM) for high-level planning and a visual model for trajectory generation as a "geometric plan." (Yang et al., 2024; Wu et al., 2024; Yang et al., 2025) generate high-level solution guidelines and retrieve them for specific problems. These guidelines are then elaborated upon to solve the target problem. Yang et al. (2025) created a library of approximately 500 generic thought templates and trained models using them. In contrast, Wang et al. (2024) first derives a high-level solution strategy and then retrieves previously generated demonstrations aligned with that strategy to solve the problem. The idea of first elaborating on task specific reasoning has also been proposed, e.g., Zhou et al. (2024); Gao et al. (2024). In Gao et al. (2024), a model first selects a reasoning method (e.g., CoT, ToT, self-refine) and then solves the problem using that method.

#### 3.1.2 Reflection

Reflection in the context of learning is "exploring one's experiences in order to lead to new understanding and appreciations" (Boud et al., 1985). AI literature has widely discussed two core concepts: (i) verification, which assesses or validates generated outputs such as the quality or truthfulness of step-by-step solutions or partial results, and (ii) refinement, which aims to improve prior outputs. A key question is what information is incorporated in the reflective process, especially regarding the availability of external feedback. A model may reflect on its output without external information (e.g., during planning), or it may use observations and direct feedback from the environment. Although reflection is considered essential to reasoning, the capacity of LLMs to reflect using simple prompts without external input has been questioned (Huang et al.,

2024; Zhang et al., 2024). Some improvements have been attributed to incorporating feedback from oracles, such as compilers (Chen et al., 2023). Verification: Verification allows to stop or redo a "chain-of-thoughts", if intermediate result wrong or deemed highly uncertain. Using multiple alternative chain-of-thoughts for verification--i.e., to ensure self-consistency--has been proposed (Wang et al., 2022a). This ensembling method improves output quality through majority voting; for example, if a model produces four answers--4, 8, 7, and 4--it would select "4" as the final result. Imani et al. (2023) focused on explicitly verifying (i) intermediate rather than final solutions, (ii) through diverse solution strategies, and (iii) using external tools as solvers. Specifically, they reformulated math problems to be solvable by both an algebraic solver and a Python interpreter. They compare the outputs of both approaches. If the outputs were not identical, further invocations were initiated. Chen et al. (2023) aimed at code error correction by using debugging samples (in-context learning) and using feedback from a compiler to assess generated code. The approach of assigning confidence scores to individual reasoning steps--rather than to the entire solution--has also proven effective (Razghandi et al., 2025). In principle, the same model that is used for generation can also be used for verification, though specialized models trained for verification can lead to better outcomes (Hosseini et al., 2024). Refinement: Progressive-hint prompting improves results iteratively by appending prior outputs to the input (Zheng et al., 2023). For instance, if the model initially answers "5," the next iteration might append "The answer is near 5" to the prompt. In the self-refine method (Madaan et al., 2023), a separate prompt is generated to collect feedback on the model's output. The feedback, along with the initial input and output, is used to produce a refined solution. This process is repeated by appending each iteration's output and feedback to the next prompt. In this way, the LLM builds a "chain of reflections" by accumulating prior outputs. This concept of iterative accumulation appears in other self-reflection studies as well (Shinn et al., 2023; Yao et al., 2023). Furthermore, guiding the reflection process through a meta-reflection process has also been proposed (Liu et al., 2025).

**Fig. 8** Evolution of reasoning from direct input output prompting, to Chain-of-thought to a forest and graph of thought (Figure enhanced with GoT from (Bi et al., 2024))

#### 3.1.3 Planning and Search

Tree of Thought (ToT) (Yao et al., 2023), Forest of Thoughts (FoT) (Bi et al., 2024), and Graph of Thoughts (GoT) (Besta et al., 2024) represent further developments of the idea of generating multiple reasoning paths, as in self-consistency (Wang et al., 2022a), which are assessed and filtered to yield improved outcomes--see Figure 8. These approaches share the view that evaluating complete chain-of-thoughts may be computationally expensive for several reasons: It is often preferable to halt solution generation early if the path appears unlikely to produce a good result. If an intermediate step is incorrect, it may be more effective to refine that specific step before proceeding with the reasoning process. Additionally, if a reasoning chain yields strong intermediate results, it may be beneficial to retain the initial steps and explore alternative continuations. Conceptually, after each reasoning step, the partial chain is evaluated to determine whether it is worth pursuing further. Such concepts are well-rooted in the classical search literature. Therefore, extensions of the Chain-ofThought (CoT) paradigm can be understood as fusions of classical search techniques with LLMs. For instance, ToT explores strategies like depth-first search--evaluating each chain fully--and breadth-first search, where multiple partial chains are expanded concurrently by choosing the next step for each, potentially growing the set of active chains. Pruning is employed to terminate the progression of partial chains that receive poor evaluations.

Assessing intermediate results: A major challenge lies in evaluating partial solutions when neither the final outcome nor the correctness of intermediate steps is known. To address this, methods like ToT use self-consistency voting (Wang et al., 2022a) or enlist an LLM as a solution evaluator (Besta et al., 2024; Bi et al., 2024). Human evaluators (Bi et al., 2024) and problem-specific heuristics can also serve in this assessment role. That is, though ideally, general methods are used, task-specific assessment methods can be more effective. As a result, such systems typically include components beyond the LLM. A controller maintains the search state, generates prompts, parses outputs, and ranks candidate next steps. While scoring and validation may be handled by the controller, they are often performed by an LLM--see the common architecture in Figure 9. In the general graph framework (Besta et al., 2024), specific prompts are used to generate subtasks (i.e., graph nodes as solution steps), solve them, assign scores, evaluate them, and merge the resulting thoughts. Alternatively, a model can be fine-tuned on search data to enhance its search capabilities (Gandhi et al., 2024) or to internalize the search process itself (Schultz et al., 2024), potentially obviating the need for external search algorithms.

Diversity: FoT and other approaches (Beeching et al., 2025; Lingam et al., 2025) aim at diversity. FoT generates variations of inputs, so that for each input a separate tree is generated. In (Lingam et al., 2025) multiple agents are responsible for diverse thoughts, which are also stored (if successful) and retrieved (Lingam et al., 2025). Beeching et al. (2025) uses beam search paired with a (domain-specific) process reward models to achieve diversity.

Planning: Similar to search, existing planning techniques and tools have been adopted (Liu et al., 2023; Hao et al., 2023). Liu et al. (2023) generates as output

**Fig. 9** Architecture for CoT combined with search with a separate controller containing also Scoring and validation.

instructions that are executed by a specific planning tool. In contrast, Hao et al. (2023) adopts Monte Carlo Tree Search (MCTS), where the LLM incrementally builds a search tree. LLMs have also been used with heuristic planning, where an LLM generates actions, evaluate their feasibility and long-term payoff leveraging learnable domain knowledge (Hazra et al., 2024). For more information on how LLMs are incorporated in planning consult the survey (Pallagani et al., 2024).

#### 3.1.4 Learning to Reason

Pretraining using next-word prediction has remained largely consistent since the early days of LLMs (Radford et al., 2019; Brown et al., 2020; Touvron et al., 2023; AI, 2025), with autoregressive decoder-only models continuing to dominate. Multimodal models leverage distinct encoders to deal with diverse data such as text and images (Touvron et al., 2024). Encoder-decoder models like T5-Flan (Chung et al., 2024) or bidirectional models like BERT (Devlin et al., 2019) have become less prevalent, though are still heavily researched, e.g., Schneider (2025) uses a bidirectional model for verification--generating next tokens via a decoder and predicting the second-last token using a verifier model.

Training to reason: The reasoning capabilities of LLMs, often elicited through prompts like "Think step-by-step" (Kojima et al., 2022), are instilled during training using datasets that include examples of step-wise reasoning. In other words, training models with next-word prediction on data containing step-wise reasoning (Brown et al., 2020) can endow them with basic reasoning skills. Therefore, an obvious next step for enhancing reasoning is to gather more reasoning-focused data and train on it. While the original CoT paradigm (Wei et al., 2022c) required users to provide example reasoning paths, an approach has since been proposed to incrementally generate CoTs and alternate between their generation and training (Zelikman et al., 2022). This approach requires a small number of demonstrations and a larger database of tasks and solutions (but without any rationale). Specifically, the LLM generates a new CoT for a given question by leveraging related existing demonstrations. If the generated solution is correct, its CoT is added to the demonstration dataset. The model is then fine-tuned on this newly expanded set of CoT demonstrations. Training on

domain-specific data also enables domain-specific reasoning, as shown for the medical domain (Zhang et al., 2023).

Training a State-of-the-art model: DeepSeek Although reasoning models generally follow the same pre-training phase, subsequent training stages may differ. After pre-training, models are typically subjected to supervised fine-tuning (SFT) using high-quality instruction datasets. Without this step, later (reinforcement learningbased) methods often struggle to meaningfully improve reasoning due to a poorly grounded base behavior. Accordingly, the DeepSeek paper (Guo et al., 2025) offers a detailed overview of a recent state-of-the-art reasoning model called DeepSeek-R1 and the earlier works that contributed to its development.1

DeepSeekMath (Shao et al., 2024) carefully curated web-data to identify data relevant for mathematical reasoning. This data was used to train models via a reinforcement learning technique known as Group Relative Policy Optimization (GRPO). GRPO evaluates behavioral policies by averaging output groups instead of computing explicit value functions or models, which are more computationally intensive. This method constituted one component in the training process of DeepSeek-R1 (Guo et al., 2025). The best performing DeepSeek model variant built upon a strong base model (DeepSeek V3). First, it was trained on a small set of high-quality CoT examples. Without this data, the model's outputs were significantly less human-readable. Second, it applied GRPO reinforcement learning using a reward function that considered: (i) solution accuracy--assessed via rules or external tools like code interpreters; (ii) formatting, to ensure output usability; and (iii) language consistency--avoiding mixed-language responses (e.g., Chinese and English), though this slightly reduced performance. The third step involved supervised fine-tuning on a mix of self-generated reasoning and non-reasoning data, along with the data used to train DeepSeek V3. Finally, to achieve human alignment (helpfulness and harmfulness) another training stage is conducted.

Incorporating human preferences: LLMs have also been trained to reflect human preferences--rather than step-wise reasoning instructions--using RL-based methods and supervised training, such as training on generated sentences conditioned by user feedback (Rafailov et al., 2023; Ziegler et al., 2019; Liu et al., 2023). Typically, humans rate two or more reasoning options, and the LLM learns to infer latent factors that capture these preferences. However, to reduce data collection costs, LLM-generated feedback has been used to emulate human responses (Dubois et al., 2023b). Nevertheless, emulated feedback must be high-quality, as data quality is considered more critical than algorithm choice for achieving strong performance (Ivison et al., 2024). While these methods are vital for improving human-LLM alignment (in terms of helpfulness and safety), they are less commonly used to enhance reasoning, though they can be applied for that purpose. When oracle feedback for the final output is available, it can be used to derive rewards for intermediate steps (Zhang et al., 2024), enabling the creation of training data for self-supervised learning and enhanced reasoning.

**Fig. 10** Retrieval augmented generation (RAG) (Lewis et al., 2020) enhances prompts with external information. It requires generation of a vector database which uses vectors summarizing text chunks as index for retrieval.

| Characteristic |  |  |
|---|---|---|
| Memory functions | Reasoning, personalization, processing large data and learning |  |
| Memory Persistence | Short-term (Working memory, model context) | Long-term (parameters, databases) |
| Memory Type | Architectural, retrieval-based, parametric, and ephemeral memory |  |
| Memory Location | Internal (parameters, context window), external (databases, APIs, accumulation of inputs/outputs), and hybrid |  |
| Information Source | Agent-discovered (feedback, reflection) | External knowledge (documents, collaboration) |
| Memory access | Non-retrieval (direct access of relevant information) | Retrieval-based (can introduce errors) |

**Table 4** Characteristics of Memory and Stored Information

### 3.2 Memory

#### 3.2.1 Memory Types and Structures

Memory and Information characteristics are summarized in Table 4. Memory functions are multifaceted. Memory is essential for processing user prompts--early models were unable to handle large prompts due to small context windows. This limitation hindered applications such as personalization and the processing of large input data, e.g., books. Memory is also relevant for reasoning as reasoning commonly requires solving problems in sub-steps, which leads to more tokens being generated than just outputting a solution. Furthermore, staying up-to-date (e.g., learning about changes and novel developments after model training, i.e., after collection of the training data) needs either updating the model parameters or access to memory containing such information. Similarly, to improve on failures requires that novel experiences are leveraged, which must be stored somewhere.

Technically, model parameters represent a form of long-term persistent memory, encoding training data via a learning process involving fitting and compression, as

1OpenAI's training process for o1 and o3 is not public.

in typical deep learning models (Schneider and Prabhushankar, 2024). Although finetuning model parameters is common, particularly during training, online updates pose risks such as catastrophic forgetting (French, 1999; Kirkpatrick et al., 2017) or malicious manipulation, as seen in real-world cases like the Tay chatbot and BlenderBot 3 (BBC News, 2016; CDOTrends, 2022). Therefore, model parameters (i.e., parametric memory) are generally treated as fixed in scenarios requiring instant learning through environmental interaction.

Parametric memory represents just one type of memory. Ephemeral memory refers to non-persistent memory used during request processing. It determines how many tokens a model can process simultaneously. Like human working memory, which is limited to about seven items (Miller, 1956), model context windows are bounded--though modern models can now hold millions of tokens (Team et al., 2024; AI, 2025). In contrast, retrieval-based memory (Lewis et al., 2020) offers virtually unlimited capacity. However, accessing memory is challenging--retrieving relevant information without including irrelevant content is non-trivial. Architectural memory refers to memory mechanisms built directly into the model architecture. For instance, MemGPT (Packer et al., 2023) introduces a memory hierarchy that mitigates context window limitations by loading from long-term memory and discarding temporarily unneeded information.

Memory location can be internal (parametric or ephemeral), external (retrievalbased), or hybrid (architectural memory). In some cases, memory consists solely of the current prompt as well as past inputs and outputs. For example, in a chat, typically information on past utterances of both the user and the model are accumulated and included throughout the chat to any user prompt. In a reflective, iterative process, a prompt might be enhanced in each iteration using past generated feedback (Madaan et al., 2023). In these cases, memory is continuously accumulated and fed to the model in full, rather than being selectively retrieved based on queries.

#### 3.2.2 Retrieval Augmentation

Retrieval augmentation involves selecting a smaller, relevant subset from a large knowledge source to include in the model's input. Knowledge retrieval reduces hallucinations, incorporates post-training or private data, and enables dynamic learning from experience. Classical RAG approaches (Lewis et al., 2020) use vector databases built by chunking large documents into smaller text snippets. Each snippet is encoded into a vector, which serves as an index for retrieval. During retrieval, the prompt is transformed into a vector to locate relevant text snippets, which are then appended to the prompt, as shown in Figure 10. Key design challenges include chunk size, vector encoding strategies, and the number of chunks to include. Several studies aim to address these challenges. For instance, Asai et al. (2023) proposes adaptive retrieval through iterative self-reflection. Fine-tuned LLMs can emit retrieval tokens to signal the need for more information. Zhang et al. (2024) fine-tunes a model to filter out irrelevant retrieved content. Yan et al. (2024) employs an evaluator to classify retrieved knowledge as correct, incorrect, or ambiguous. Incorrect knowledge prompts further retrieval, while correct knowledge is decomposed, filtered, and recomposed before being passed to the LLM. Vector databases can hold external information but can also be

used to store and retrieve experiences of an agent interacting with an environment, e.g., Zhao et al. (2024b) stores experiences but also further processes them using an LLM to learn more abstract insights. Experiences may include irrelevant details that hinder retrieval efficiency and unnecessarily occupy the context window (Zheng et al., 2023). MemGPT (Packer et al., 2023) proposes a memory hierarchy, which allows to overcome the limited context window size by loading from long-term memory and discarding information not temporarily needed. This architecture is inspired by memory management techniques used in operating systems. Furthermore, there is also work that deals with GPU memory management to improve throughput, i.e., by maintaining past computations (key-value pairs) during the next token generation process (Kwon et al., 2023). The integration of search into the reasoning process without labeled data and GPRO has been successfully performed (Chen et al., 2025).

Models with large context sizes often allow to forego retrieving subsets of the data in favor of including the entire information as part of the input. Relying on context only has been shown to lead to better performance at the price of higher generation costs and trade-offs based on dynamic decisions making have been proposed (Li et al., 2024).

Retrieval from classical databases and knowledge graphs Aside from vector databases, LLMs can also interact with classical relational databases by generating SQL queries (Li et al., 2023). Furthermore, classical word based indexing methods like BM25 (Karpukhin et al., 2020) can be used as well, which are good for exact matches (e.g., of rare words), speed, interpretability, and simplicity. Rather than retrieving from databases the idea of retrieving from knowledge graphs has gained a lot of traction as surveyed in (Zhang et al., 2025). Graphs allow to better decompose and interconnect information. For example, graphs allow to easily represent hierarchies of knowledge. Moreover, several works have developed specialized agents for RAG tasks (Singh et al., 2025), such as agents tailored for SQL databases and real-time web data retrieval.

### 3.3 Tools

Tools are external functionalities that can be invoked by an agent. They complement agents and offer several benefits: (i) Reducing errors and increasing efficiency--tools can improve accuracy and speed, such as using a calculator for basic math operations. (ii) Functionality and interaction--tools enable new capabilities for task-solving and engagement; for instance, a web browser allows an agent to perform tasks like online shopping. (iii) Interpretability and control--tools built using classical software can embed rules in a clear and reliable way, enhancing transparency and oversight.

#### 3.3.1 Tool Creation

Since LLMs can generate code, they are also capable of creating their own tools. The concept of formal tool creation by LLMs has been explored in several works (W¨olflein et al., 2025; Yuan et al., 2023; Cai et al., 2023). W¨olflein et al. (2025) converts academic papers containing code into tools accessible by LLMs, while (Cai et al., 2023; Ding et al., 2025) generates Python utility functions as tools. In addition to generating

tools with GPT-4, the process in (Cai et al., 2023) includes proposing tools using three training examples, verifying them through unit tests, and wrapping them with documentation and examples. Similarly, Richards (2023) supports dynamic creation and execution of scripts by agents.

#### 3.3.2 Tool Selection

An agent might be able to handle multiple tools. Different tools support diverse solution strategies. As a result, selecting the appropriate tool becomes an integral part of the planning process (Ruan et al., 2023). Tool selection can be addressed by adapting the chain-of-thought paradigm (Chen et al., 2023). Schick et al. (2023) employed self-supervised learning to determine optimal tool invocation times. ReActstyle prompting (Yao et al., 2023) integrates reasoning and action, encompassing tool selection as part of the agent's decision-making.

#### 3.3.3 Tool Use

Schick et al. (2023); Parisi et al. (2022a) utilize software tools to carry out tasks. To this end, they fine-tune an LLM to generate API calls to invoke software and parse its responses covering simple tools such as a calculator, translation system and search engine. Subsequent research has focused on enhancing fine-tuning and prompting to improve API call accuracy (Patil et al., 2024), including support for "out-of-distribution" APIs not explicitly optimized during training (Qin et al., 2023). Self-play approaches have been proposed (Parisi et al., 2022b), where a small set of tool usage examples is iteratively expanded by adding sampled usages that yield reasonably good outputs. The use of programming language interpreters has become increasingly common. The LLM can output executable code that is run by a program interpreter to answer a prompt, as demonstrated in (Gao et al., 2023). Some approaches (Wang et al., 2024; Qiao et al., 2023) implement reasoning through code, refining it step-bystep in response to environmental feedback. LLMs are also capable of generating SQL queries to handle analytical tasks using relational databases (Li et al., 2023). LLMs themselves may be treated as tools, as shown in (Shen et al., 2023), where tasks are decomposed and subtasks routed to specialized LLMs.

### 3.4 Interacting

#### 3.4.1 Reinforcement Learning

The concept of interaction and agents has long been central to AI, as reflected in the definition: agents "receive percepts from the environment and perform actions" (Russell and Norvig, 2021). In reinforcement learning (RL), agents receive rewards--sometimes in the form of human feedback--that signal the quality of their behavior (Russell and Norvig, 2021), as illustrated in Figure 5. Agentic AI incorporates RL by leveraging foundation models to apply extensive world knowledge. For instance, Voyager (Wang et al., 2023) demonstrates continual learning in Minecraft through exploration, building a skill library, and iterative prompting that incorporates feedback, execution errors, and self-verification.

World model and policy: An RL agent simultaneously learns a world model and a behavioral policy. The world model defines possible environmental states, the agent's state, and transitions between states. The policy dictates which action an agent should take in a given state. Both the world model and the policy are learned from past experiences and received feedback.

Exploration and Exploitation: Unlike classical supervised or unsupervised learning where data is given, RL agents actively influence data collection through novel actions and observations--i.e., exploration. Agents must balance exploration (acquiring knowledge) with exploitation (applying existing knowledge to perform tasks). RL agents typically execute a sequence of actions to reach goals, each involving perception, reasoning, and action.

RL and Agentic AI:There is no one-to-one correspondence between RL agents and agentic AI systems. Agentic AI targets broader, more open-ended tasks and may define its own objectives. An RL agent aims to optimize a (more narrow) reward function by learning a policy for how to act in a specific environment. Nonetheless, Agentic AI incorporates many RL concepts, especially during training (Guo et al., 2025). RL architectures like actor-critic models--which separate decision-making (actor) from evaluation (critic), often via self-assessment--are also commonly adopted in Agentic AI systems.

**Fig. 11** Instruction to (Anthropic's) agent controlling a computer

#### 3.4.2 Interacting in (Virtual) Worlds

Often agents operate in restricted or digital environments though attempts have been made to consider general environments moving towards models that consider the world in its entirety or at least narrow tasks involving the physical world (or a model of it). For example, agents have also been employed in specialized simulation environments for robotics (Wang et al., 2024).

**Fig. 12** (Anthropic's) agent controlling a computer through mouse, keyboard and sensing through screenshots as illustrated in the left panel

Multi-modal models that are trained on text, images and videos might constitute a potential world model (Ge et al., 2024). That is, such models should, for example, given an image of the current state (such as a plane) and a textual description of an action (the plane will land soon) predict the next state (showing a plane that has landed). Xiang et al. (2023) deploys an agent in a model of a virtual home, which simulates the physical world. The agent performs random exploration and goaloriented planning to gain experience, which are used to fine-tuning a LLM. The finetuned LLMs perform significantly better than much larger non-fine-tuned LLMs. Due to their extensive knowledge LLMs can be used as both a world model and a reasoning agent engaging in planning (Hao et al., 2023). Here the reasoning agent builds a reasoning tree suggesting actions to solve a task under the guidance of the world model providing rewards.

While ordinary generative vision-language models allow to textually describe images and generate images from text. Agentic AI seeks to extend to infer actions given images and text. In particular, Brohan et al. (2023) developed a vision-language-action model by training a vision-language model on trajectories of robots containing both their observations and actions visual language tasks such as visual question answering.

Motion planing and 3D: While such approaches might improve significantly on prior work for the considered scenario, they still fall short in many aspects and tasks that are innate for humans. For example, they cannot perform 3D spatial reasoning such as estimating distances or size differences between objects without extensive training on this task (Chen et al., 2024).

The idea to leverage LLMs for motion planning, i.e., planning driving trajectories has also been explored (Mao et al., 2023). They described a visual scene in text by specifying objects and their coordinates, which allowed a language model to process and reason upon the geometrical problem and output the motion trajectory.

Agents in computer games: Agents are commonly evaluated in computer games as they can be seen as open-worlds. Nottingham et al. (2023) uses two phases to play the strategy game "MineCraft": i) an LLM to plan the behavior of an RL agent by

defining subgoals, ii) an RL agent learning a policy for each subgoal and updating the LLM's world model (Wang et al., 2023) proposed an interactive planning approach for MineCraft based on describe, explain, plan, and select to yield feasible plans and reduce errors through self-explanation.

Interacting with browsers and computers The idea to control a computer or at least the browser has gained increased attention. Commercial companies like OpenAI and Anthropic (Anthropic, 2024) have already released agents performing simple tasks as illustrated in Figure 11, where the user prompt is addressed by controlling the mouse, keyboard and taking screenshots 12. Furthermore, a number of benchmarks where agents are supposed to solve tasks. To this end, multi-modal models can be used to extract information relevant to control the system from screenshots, e.g., as done with Omniparser V2 (Yu et al., 2025).

A web-browser can be seen as a tool that provides access to web-services but also as an environment to solve tasks. Systems with reasoning and planning capabilities can be enhanced with acting capabilities, e.g., Zhou et al. (2023), to make use of web-browsers (Drouin et al., 2024; Zhou et al., 2023; Boisvert et al., 2024). However, as of now performing "relatively" simple tasks for humans, e.g., related to online shopping remains a challenge for AI agents (Jin et al., 2024). More concretely, on the Workarena++ benchmark (Boisvert et al., 2024) humans score close to 100% while state-of-the-art LLMs such GPT-4o and LLama3 score close to 0%. An example task is shown in Figure 13.

**Fig. 13** Task from Workarena++ (Boisvert et al., 2024). Given a ticket asking to refill inventory below four items (top panel), the system must understand a chart to identify items to order(left lower panel) and interact with an ordering page (right lower panel).

While the idea of agents controlling a computer with all its software and any further tuning towards agent usage is appealing, designing specialized agent friendly interfaces can improve an agent's success rate as demonstrated for software engineering (Yang et al., 2024).

Interacting with tools Conceptually, tool usage (and access to external memory like databases) could also be viewed as environmental interaction. However, we treat both as integral components of the Agentic system. First, memory and tool usage often do not require a long sequence repeating perceiving, reasoning, acting, but rather tool usage is a single API call, e.g., to a database. Tool usage is often learnt in a supervised manner, e.g., by using samples of problem descriptions and the corresponding API calls. Second, we view an agent as being equipped with at least basic capabilities to use a tool. Reasoning might be needed to apply them successfully for a task if no supervised fine-tuning is performed. But if an agent lacks knowledge on tool usage, an agent can often rely on documentation, usage samples, and obtain immediate feedback, which tends to limit the needed interaction. However, especially for complex tools where no training data exists, a reinforcement learning setting might be the better conceptual match.

#### 3.4.3 Interacting with Humans

Humans can assist, collaborate, or oversee the agent. Scientific Processes: The AI scientist (Lu et al., 2024) and follow-up works (Gridach et al., 2025) implement an end-to-end LLM driven scientific discovery process including idea generation, experiment iteration and paper write-up showing that it can generate 100s of medium-quality paper within a week. Deep Research by OpenAI and Google also support researchers by generating research reports, e.g., searching and summarizing works from a particular area or about a particular topic using multiple rounds of search and analysis. In particular, these commercial tools also often ask humans for clarifications as the research process takes a significant amount of time due to a large computational demand. In the academic world, human-agent collaboration for research has been discussed in (Ifargan et al., 2025). More specialized research agents have been proposed, e.g., for drug discovery (Liu et al., 2024). Specialized agents can perform highly domain specific tasks, such as simulating molecules relevant to chemistry and modeling ecosystems as needed in biology (Cheng et al., 2024).

Agents as assistants: Agents can be used to support the execution of tasks but also for improving skills, e.g., practicing negotiations (Schneider et al., 2023). Despite an extensive number of already deployed applications, the optimal design of agents engaging with humans is still subject to research. For instance, recently back-channeling as an active listening strategy, where the LLM would utter at appropriate times phrases like "really?" or "Wow" led to higher conversational engagement (Jang et al., 2024).

Many areas aside from the aforementioned "scientific discovery" such as law, finance, psychology, education, medicine and military also benefit from Agentic AI with numerous applications (Cheng et al., 2024). For example, an agentic AI workflow has been proposed to translate formal medical reports into patient-accessible reports

reducing errors and hallucination through reflection (Sudarshan et al., 2024). Counseling agents for students being bullied have also been assessed (Paul et al., 2024) mainly by comparing existing commercial LLMs. For more medical examples consult (Wang et al., 2025). Agentic AI has also been suggested as ethical counsel in the practice of law (O'Grady and OG, 2024). In finance, multi-agent systems have been proposed for decision making (Yu et al., 2024) as well as for trading (Xiao et al., 2024).

#### 3.4.4 Interacting with Other Agents

Agents can assume various roles: autonomously simulating aspects of society or systems, acting as integrated analytical components like critics or evaluators within reasoning processes, or collaborating independently to solve complex tasks through coordinated efforts. Agents for Simulation: Agents have been used for societal and economic simulations, such as modeling macroeconomic dynamics with diverse interacting agents (Li et al., 2024). Agents have also been employed to simulate disease spread, such as during the COVID-19 pandemic (Williams et al., 2023). In recommender systems, agents simulate both users and items to model interactions and improve recommendation quality (Zhang et al., 2024a). Agents have also represented countries in simulations of historical wars (Hua et al., 2023). Agents as Analytical Components: Having two LLM agents debate under the moderation of an LLM judge has fostered more divergent thinking (Liang et al., 2023) and improved summarization (Chan et al., 2023). However, agents have also shown to reduce diversity by converging towards human-like polarization (Piao et al., 2025). Additionally, incorrect or manipulated knowledge can spread rapidly within agent networks (Ju et al., 2024). Agents as Independent Collaborators: To solve user-defined tasks, Li et al. (2023) proposed using role-playing agents, such as a Python programmer and a stock trader collaborating to develop a stock trading bot. Although these agents collaborate autonomously, the paper suggests introducing critics to provide feedback and improve outcomes. The paper also emphasizes the importance of carefully crafted initial prompts and precise task specifications, potentially generated by an LLM. Hao et al. (2025) proposes using a hierarchical structure of agents led by a central leader. In each layer i, messages from previous layers are aggregated, producing an averaging effect that stabilizes responses. The leader provides feedback that subordinate agents use for improvement.

### 3.5 Specifying and Evaluating Agentic AI

A critical aspect of any system is its specification and ensuring that the system conforms to the specification and broader goals such as legal compliance and efficiency.

#### 3.5.1 Specifying an Agent

Defining an agent typically involves describing identity information (e.g., name, age, personality), motivational drivers, professional roles (Park et al., 2023), tool permissions, delegation rights, workflows, and interaction behavior with other agents--see

Figure 15. Although agent descriptions can be elaborate, simple descriptions often suffice--for instance, in a multi-agent programming setting, agents were defined using only a professional role, a goal, and constraints (Hong et al., 2023). The definition also depends on the intended level of autonomy 2; agents with partial autonomy--whether due to technical limitations or legal reasons--may require consent checkpoints where humans review outcomes and planned future actions (Singh, 2022).

Agent Design: Manual and Automatic (Data-Driven or Model-Based): Agents can be designed either manually or automatically. Automatic design can involve fitting to data or through model optimization involving, e.g., an evaluation function. Hu et al. (2024) evaluates the automated design of agentic systems. This approach requires specifying a search space of potential agents, a search algorithm to explore the space, and an evaluation function to assess the quality of agents. A metaagent programs new agents by generating code, potentially based on archived prior agents, and employs a reflection process to ensure novelty. Evolutionary algorithms using cross-over and mutation have also been used for automatic agent creation (Yuan et al., 2024). Data fitting can be achieved using few-shot prompting with examples or through sampling techniques. For example, to emulate human participants in a user study, agent profiles can be generated by sampling traits from a distribution (Argyle et al., 2023). Wang et al. (2023) manually defined a few agent profiles, which were subsequently expanded into a larger agent pool with the assistance of an LLM.

**Fig. 14** Specifying a multi agents system using agents, tasks and their interconnection as abstractions specified a user (Figure from (Bi et al., 2024))

#### 3.5.2 Specifying Multi-Agents

Agentic systems often comprise multiple agents. Using agents with different personas--potentially based on the same underlying LLM--can achieve better outcomes than chain-of-thought (CoT) prompting for tasks like creative writing and trivia (Wang et al., 2023), motivating the adoption of multiple agents. Multi-agent systems can be defined in various ways but generally follow similar structural patterns. Agents and tasks are defined independently, allowing flexible assignment of agents to tasks, as illustrated in Figure 14. Both agents and tasks can be flexibly specified through textual descriptions, similar to conventional prompting. An agent description mirrors that of

**Fig. 15** Example definition of a content planner agent, the content planning task and the writer agent providing procedural guidance including interaction with other agents (Figure from (Bi et al., 2024))

a single-agent system, specifying goals, a role (or persona) characterizing attributes (e.g., empathetic behavior), allowed tool usage and delegation, and procedural task guidance. A task description may include expected outputs and procedural guidance, including instructions for agent interactions, as shown in Figure 14.

Multi-agent systems form distributed architectures where communication patterns and (self-)organization strategies, such as symmetry breaking (Barenboim et al., 2016), are crucial.

Key dimensions of Multi-Agents System We provide essential dimensions for multi-agents systems synthesized from works that provide a more detailed discussion (e.g., Cheng et al. (2024); Guo et al. (2024); Li et al. (2024))

• Homogeneous vs. Heterogeneous: Homogeneous agents can be exact replicas, meaning the underlying foundation model as well as the agent description are identical. Heterogeneous models can be based on different models, subject to different agent instructions. • Cooperative vs. Non-cooperative: Cooperative agents pursue a shared goal and may use argumentative debate styles to enhance results (Du et al., 2023a). • Communication structure: Communication can be classified as push--where messages are sent unsolicited--or pull--where agents request information explicitly. Communication can also be centralized--through a single agent or shared message pool (Hong et al., 2023)--or decentralized, where agents interact freely without a central node. • Organization (Flat vs. Hierarchical): Agents can be organized in hierarchical settings, e.g., Hao et al. (2025), or in a flat manner. For example, Zhang et al.

(2024) employs multiple workers handling parts of a task (sequentially) and a manager aggregating results. • Learning: Agents may learn independently in decentralized systems or share experiences centrally to accelerate collective improvement. • Communication content: Communication is primarily text-based; cooperative systems often package instructions, goals, state descriptions, action histories, and dialogue histories into messages (Zhang et al., 2023). Agentic systems often mimic real-world development by featuring domain-expert agents that collaborate through discussions, potentially adopting varied communication styles (Yamamoto et al., 2025).

#### 3.5.3 Evaluating Agentic AI

Evaluating Agentic AI is challenging. Agents can behave non-deterministic with complex interactions and require a much longer time-span to complete tasks than GenAI. Agents can exhibit a wide range of capabilities and desiderata that require evaluation. We outline key decision domains but refer readers to specialized surveys for further details (Yehudai et al., 2025). Capabilities span general LLM skills like long-text understanding and instruction following, as well as agent-specific skills such as planning, learning from interactions, error handling, tool usage, and spatial reasoning (Wu et al., 2023). Desiderata include task performance (Wu et al., 2023), failure awareness (recognizing impending failure), efficient tool usage, harmfulness (Andriushchenko et al., 2024), and computational efficiency--for instance, ARC Prize Foundation (2025) enforced efficiency by setting computational limits for tasks. Beyond foundational capabilities, agents can also be evaluated on domain-specific competencies. For instance, evaluating agents simulating humans in user studies has involved manual comparisons with real users and sanity checks that vary names and gender to account for prompt sensitivity and model shortcomings (Aher et al., 2023). Metrics can include generic measures such as completion rate (fraction of tasks completed) or task success rate (Wang et al., 2023), possibly supplemented by solution quality scores, self-aware failure rates (fraction of tasks where agents signal failure before failing), and tool call accuracy. In games, evaluation scores could reflect deviations from optimal actions (e.g., in rock-paper-scissors) or achievements awarded by the game engine (Wu et al., 2023). Benchmarks designed for Generative AI, such as the well-known MMLU benchmark (Hendrycks et al., 2021b), can also be applied to Agentic AI; MMLU has been extended with more complex questions to better evaluate reasoning abilities (Wang et al., 2024). Various datasets exist for evaluating (or training) Agentic AI, often focusing on specific areas such as tool usage (over 14,000 REST APIs (Qin et al., 2023)), robot learning (Walke et al., 2023), assistant-user task pairs (Li et al., 2023), programming (Hendrycks et al., 2021a), social game playing (Akata et al., 2023), and simulating human behavior in user studies (Aher et al., 2023). Frameworks are also available to facilitate task evaluation. Lin et al. (2023) offers infrastructure enabling researchers to easily design custom evaluation tasks. Games are frequently used for evaluation, ranging from simple games like rock-paper-scissors to complex environments like Minecraft (Wu et al., 2023). Several benchmarks target evaluating agents

in web environments (Drouin et al., 2024; Pan et al., 2024), focusing on completing basic human tasks (Drouin et al., 2024).

## 4 Challenges Toward AGI

### 4.1 Challenges

Many challenges found in GenAI also apply to Agentic AI, though some become either more pronounced or mitigated. They hinder the further evolution to Artificial General Intelligence (AGI). These challenges also posit research opportunities.

Errors: Although reasoning reduces errors compared to Generative AI, as shown in benchmarks (Figure 4), the complexity and length of tasks in Agentic AI increase the risk of cumulative errors across steps (Shavit et al., 2023).

Interpretability: Agentic AI emphasizes step-by-step reasoning presented in readable text, enhancing interpretability. However, Chain-of-Thought (CoT) reasoning may not always be faithful to the underlying decision process (Turpin et al., 2023). Moreover, GenAI (Schneider, 2024a) and deep learning broadly (Longo et al., 2024) continue to struggle with explainability. Additionally, Agentic AI is more complex than GenAI because it integrates components like planning algorithms, external memory, and tool usage.

Dynamic and Complex Environments: Agents are designed to operate in dynamic, complex environments. They interact with changing environments, select tools, and dynamically collaborate with other agents, sometimes in parallel.

Observability: The environment might not be fully observable, e.g., we cannot gather all information about our world. This is a common assumption in RL (Russell and Norvig, 2021). Furthermore, agents' behaviors and tool operations may not be fully transparent. For example, agents might access tools via APIs whose internal mechanisms are opaque.

Safety and Security: Agents are vulnerable to harmful behaviors, either through deliberate attacks (Andriushchenko et al., 2024) or systemic shortcomings. This risk is higher for Agentic AI systems compared to GenAI, due to their increased interaction in less controlled environments. For instance, agent capabilities like tool invocation can be exploited. Furthermore, multi-modal agents also face attacks on each modality, e.g., foundation models can be jailbroken through adversarial images (Qi et al., 2024). Privacy concerns escalate as agents share information with other agents and tools. Even seemingly minor information--such as an agent's role as a psychological counselor--can represent a privacy breach.

Evaluation: Evaluating Agentic AI--assessing reliability, task performance, and potential harms--remains challenging and incomplete (Shavit et al., 2023).

Human Alignment: Agents may engage in unforeseen or unethical actions in pursuit of their goals, contrary to human intent.

Controlling and Monitoring Agents: Controlling agents is challenging. Even when human or independent approvals are required for agent actions, anticipating the consequences of approvals or disapprovals remains difficult (Shavit et al., 2023). Because agents are dynamic, they may adopt unpredictable behaviors to achieve their goals.

Resource Allocation and Management: Managing resources, such as computational access, is more demanding as agents dynamically consume varying and potentially arbitrary amounts of information (Cheng et al., 2024).

### 4.2 Agentic AI to AGI?

AGI refers to AI that transcends narrow domains and can generalize to new, unfamiliar situations. Current AGI benchmarks, such as (ARC Prize Foundation, 2025), emphasize novel tasks requiring learning from few examples and limited computation, rendering brute-force approaches impractical. Agentic AI enables novel applications beyond those possible with GenAI. Tasks requiring more complex reasoning and interactions become more and more feasible. Currently, agents are restricted to relatively simple tasks due to technological and regulatory constraints, but this is expected to evolve. Agentic AI may represent the next step toward Artificial General Intelligence (AGI), potentially equaling or surpassing human intelligence. Incremental technological advances and scaling of existing Agentic AI systems might be sufficient to achieve AGI. However, progress may stall due to barriers like limited availability of training data (Mok, 2025). Some researchers argue that mere scaling is insufficient, and fundamentally new approaches are required (Bengio and Hu, 2023). Consequently, the timeline for achieving AGI remains uncertain, as evidenced by the wide range of predictions from respected researchers. For instance, in 2018, Ray Kurzweil predicted a 50% chance of human-level AI by 2029, while Stuart Russell estimated 50-70 years, Yann LeCun 50-100 years, and Rodney Brooks approximately 180 years (Ford, 2018). What happens if AGI is reached? Even decades before the current AI boom, there were speculations about the consequences of achieving AGI (Kurzweil, 2005; Bostrom, 2014). It is arguably difficult to predict the outcomes once AI surpasses human intelligence. Researchers like Kurt Russell suggest that once AI reaches a "Kindergarten level," its improvement could accelerate at more than an exponential rate (Ford, 2018). In particular, AGI has been said to posit an existential risk to humanity (Bostrom, 2014) posing challenges with respect to control and value alignment. According to the instrumental convergence thesis, AI systems might independently pursue goals like selfpreservation and resource acquisition, potentially clashing with human interests. These existential risks go far beyond more tangible economic risks such as unemployment and potential erosion of wages (Bostrom, 2014).

## 5 Methodology and Related Work

### 5.1 Methodology and Scope

Scope and Target Audience: Our focus is on contrasting Agentic AI and GenAI and highlighting key innovations in the transition, targeting a broad audience of academics and industry professionals. The basic characteristics distinguishing GenAI from

Agentic AI are intended to be accessible to a wide audience, while recent innovations in Agentic AI are tailored for a more technically focused readership. To this end, we adopt a high-level capability perspective aimed at broad interest. We assume readers have basic familiarity with deep learning and Generative AI concepts, including foundation models and prompt engineering, as covered in prior works (e.g., Schneider et al. (2024)). In-depth technical knowledge, such as of the transformer architecture, is not our focus, as it is covered extensively in other works (e.g., Schneider (2024b)). While we address all relevant areas of Agentic AI, we refer readers to other works for more extensive coverage of specific aspects. Research Methodology: We primarily followed the literature review methodology outlined by (Wohlin, 2014), incorporating several innovations to address the rapid growth of research in this field. Specifically, we began by conducting a meta-survey, searching Google Scholar for surveys on "Agentic AI," "LLM agents," and "Generative AI." This approach served three goals: (i) identifying key papers for forward and backward search (i.e., snowballing (Wohlin, 2014)), (ii) building on existing works to ensure conceptual completeness, and (iii) improving prior works by synthesizing different viewpoints and identifying gaps. We assessed more than 30 surveys, but identified five surveys on Agentic AI (Wang et al., 2024; Xi et al., 2025; Acharya et al., 2025; Plaat et al., 2025; Cheng et al., 2024) and two on Generative AI (Zhang et al., 2023; Manduchi et al., 2024), which were chosen due to quality, recency, comprehensive and non-overlap with other chosen surveys. The Agentic AI surveys were significantly more influential in shaping our work. We conducted an initial read-through of these manuscripts. We then developed an outline for our paper, including chapter structures and basic content, unifying prior works while omitting less relevant aspects and introducing novel ones. In a second iteration, we revisited the surveys to identify gaps in our conceptualization, refining our framework primarily through backward search. In a third iteration, we conducted forward searches, focusing on citations and publications in top venues like NIPS, ICLR, and ICML, while also considering new arxiv.org papers. We restricted our search to works from 2024 onward, as older works were assumed to be captured in the selected surveys. Additionally, we performed targeted searches on Google Scholar for each subchapter, emphasizing recent surveys on topics like evaluation, reasoning, planning, and RAG. These sources were used to further refine the structure and content of our work. To ensure the inclusion of the most recent results, we incorporated works from arxiv.org and selected blog posts from reputable sources such as OpenAI and Huggingface. However, these additional sources were included only after basic quality assessments.

### 5.2 Related Surveys

As stated in the research methodology, we relate most strongly five surveys on "Agentic AI" and "LLM Agents". Key differentiators include: (i) None of the surveys focused on distinguishing GenAI from Agentic AI, which forms the first part of our survey. (ii) None offered a clear and multi-angled definition of Agentic AI. (iii) None discussed autonomy levels and the motivation for Agentic AI--including a contrasting perspective to AGI--in comparable detail. The second part of our survey overlaps more with existing works, but we expand on important practical aspects such as (iv) defining

agents and memory characteristics, and (v) offer a different conceptual framework. We believe that offering multiple, diverse perspectives on Agentic AI is highly valuable. We now discuss these surveys in greater depth and highlight how they differ from our work:

Wang et al. (2024), surveying "LLM-based autonomous agents," adopts an architecture-centric view describing key components but lacks a detailed discussion on defining agents. Our work is more capability-focused and differs significantly in its conceptualization. Additionally, we emphasize distinguishing GenAI from Agentic AI.

Xi et al. (2025) adopts a conceptualization emphasizing brain, perception, and action, whereas we focus on reasoning and interaction. They explore the historical notion of agents and the motivation for using LLMs, while we emphasize the differentiation between GenAI and Agentic AI.

Acharya et al. (2025) surveys "Agentic AI" by comparing it to traditional AI, including early rule-based systems. In our view, the traditional AI era began in the 1960s and ended with the breakthroughs of Generative AI, notably GPT-2/3 around 2019. While their comparison is highly valuable, it is of less interest towards readers already familiar with GenAI such as ChatGPT. Our survey targets these readers by starting with Generative AI and not covering the entire history of AI. As a result, our conceptualization differs significantly--for example, Table 1 in (Acharya et al., 2025) and our Table 1 share only one common aspect.

Plaat et al. (2025) surveys "Agentic LLMs." Our work differs by contrasting Agentic AI directly against GenAI. Our conceptualization also differs: at the highest level, we emphasize two rather than three core capabilities; additionally, we view retrieval augmentation not as reasoning (as in Figure 3 of (Plaat et al., 2025)), but rather as interaction with a tool, aligning with earlier works like (Lewis et al., 2020). Basic reasoning forms like chain-of-thought (Wei et al., 2022c) do not require retrieval, and even more advanced reasoning typically only necessitates large context windows. We offer a broader set of motivations for Agentic AI (compare Table 3 in our work to Chapter 1.4 in (Plaat et al., 2025)) and discuss agent definitions in greater depth.

Cheng et al. (2024) surveys "LLM-based agents" by comparing RL and LLM agents, whereas we contrast Agentic AI against GenAI. Our conceptualization differs at multiple levels: at a high level, we focus on novel AI agent capabilities like reasoning and interaction, while (Cheng et al., 2024) adheres closely to traditional RL agent concepts.

## 6 Conclusion

Agentic AI is a major paradigm shift beyond Generative AI by introducing reasoning, interaction, and autonomy at a new scale. Our survey and conceptualization systematically contrasts Agentic AI and GenAI from multiple perspectives. We cover its technical foundations, practical specification, and open challenges. As Agentic AI evolves, understanding its capabilities, risks, and the nuances of agent specification becomes essential for both advancing research and ensuring responsible deployment.

## Declarations

### Conflict of Interest

The authors declare that they have no conflict of interest.

### Ethical Approval

Not applicable.

### Informed Consent

Not applicable.

### Funding

No funding was received for conducting this study.

### Data Availability Statement

Data sharing is not applicable to this article as no datasets were generated or analyzed during the current study. All references are included in the paper.

### Author Contributions

Not applicable. There is just one, which did everything.

### Acknowledgments

Not applicable.

## References

Aher, G.V., Arriaga, R.I., Kalai, A.T.: Using large language models to simulate multiple humans and replicate human subject studies. In: International Conference on Machine Learning, pp. 337-371 (2023). PMLR

Argyle, L.P., Busby, E.C., Fulda, N., Gubler, J.R., Rytting, C., Wingate, D.: Out of one, many: Using language models to simulate human samples. Political Analysis 31(3), 337-351 (2023)

Agent.ai: AI Agents Explained. https://docs.agent.ai/ai-agents-explained. Accessed: 2025-04-24 (2025)

Ajay, A., Han, S., Du, Y., Li, S., Gupta, A., Jaakkola, T., Tenenbaum, J., Kaelbling, L., Srivastava, A., Agrawal, P.: Compositional foundation models for hierarchical planning. Advances in Neural Information Processing Systems 36, 22304-22325 (2023)

AI, M.: Llama 4: Multimodal Intelligence. Accessed: 2025-04-23 (2025). https://ai. meta.com/blog/llama-4-multimodal-intelligence/

Acharya, D.B., Kuppan, K., Divya, B.: Agentic ai: Autonomous intelligence for complex goals-a comprehensive survey. IEEE Access (2025)

Anthropic: Introducing Computer Use, a New Claude 3.5 Sonnet, and Claude 3.5 Haiku. Accessed: 2025-04-09. https://www.anthropic.com/news/ 3-5-models-and-computer-use ARC Prize Foundation: ARC-AGI-2: Abstraction and Reasoning Corpus for Artificial General Intelligence v2. GitHub. Accessed: 2025-04-16 (2025)

Akata, E., Schulz, L., Coda-Forno, J., Oh, S.J., Bethge, M., Schulz, E.: Playing repeated games with large language models. arXiv preprint arXiv:2305.16867 (2023)

Andriushchenko, M., Souly, A., Dziemian, M., Duenas, D., Lin, M., Wang, J.,

Hendrycks, D., Zou, A., Kolter, Z., Fredrikson, M., et al.: Agentharm: A benchmark for measuring harmfulness of llm agents. arXiv preprint arXiv:2410.09024 (2024)

Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.: Self-rag: Learning to retrieve, generate, and critique through self-reflection. In: The Twelfth International Conference on Learning Representations (2023) BBC News: Microsoft Chatbot Is Taken Offline After It Learns to Be Racist. Accessed: 2025-04-18. https://www.bbc.com/news/technology-35902104

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., et al.: Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818 (2023)

Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Podstawski, M., Gianinazzi, L.,

Gajda, J., Lehmann, T., Niewiadomski, H., Nyczyk, P., et al.: Graph of thoughts: Solving elaborate problems with large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 17682-17690 (2024)

Barenboim, L., Elkin, M., Pettie, S., Schneider, J.: The locality of distributed symmetry breaking. Journal of the ACM (JACM) 63(3), 1-45 (2016)

Bengio, Y., Hu, E.J.: Scaling in the Service of Reasoning & Model-Based ML. https://yoshuabengio.org/2023/03/21/ scaling-in-the-service-of-reasoning-model-based-ml/. Accessed: 2025-04-23 (2023)

Bi, Z., Han, K., Liu, C., Tang, Y., Wang, Y.: Forest-of-thought: Scaling test-time compute for enhancing llm reasoning. arXiv preprint arXiv:2412.09078 (2024)

Boud, D., Keogh, R., Walker, D.: Reflection: Turning Experience Into Learning. Routledge, ??? (1985)

Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., Amodei, D.: Language models are few-shot learners. arXiv preprint arXiv:2005.14165 (2020)

Bostrom, N.: Superintelligence: Paths, Dangers, Strategies. Oxford University Press, Oxford, UK (2014)

Boisvert, L., Thakkar, M., Gasse, M., Caccia, M., Chezelles, T., Cappart, Q.,

Chapados, N., Lacoste, A., Drouin, A.: Workarena++: Towards compositional planning and reasoning-based common knowledge work tasks. Advances in Neural Information Processing Systems 37, 5996-6051 (2024)

Beeching, E., Tunstall, L., Rush, S.: Scaling test-time compute with open models (2025). https://huggingface.co/spaces/HuggingFaceH4/ blogpost-scaling-test-time-compute

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., Tafjord,

O.: Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457 (2018)

Chan, C.-M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., Liu, Z.: Chateval: Towards better llm-based evaluators through multi-agent debate. arXiv preprint arXiv:2308.07201 (2023)

CDOTrends: Meta ai's new chatbot goes 'bad' in days. CDOTrends (2022)

Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X.,

Dehghani, M., Brahma, S., et al.: Scaling instruction-finetuned language models. Journal of Machine Learning Research 25(70), 1-53 (2024)

Chen, M., Li, T., Sun, H., Zhou, Y., Zhu, C., Wang, H., Pan, J.Z., Zhang, W., Chen, H.,

Yang, F., Zhou, Z., Chen, W.: Research: Learning to reason with search for llms via reinforcement learning. arXiv preprint arXiv:2503.19470 (2025) arXiv:2503.19470 [cs.AI]

Chen, X., Lin, M., Sch¨arli, N., Zhou, D.: Teaching large language models to self-debug. arXiv preprint arXiv:2304.05128 (2023)

Chen, W., Ma, X., Wang, X., Cohen, W.W.: Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588 (2022)

Chen, A., Song, Y., Zhu, W., Chen, K., Yang, M., Zhao, T., et al.: Evaluating o1-like llms: Unlocking reasoning for translation through comprehensive analysis. arXiv preprint arXiv:2502.11544 (2025)

Cai, T., Wang, X., Ma, T., Chen, X., Zhou, D.: Large language models as tool makers. arXiv preprint arXiv:2305.17126 (2023)

Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., Xia, F.: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14455-14465 (2024)

Chen, Z., Zhou, K., Zhang, B., Gong, Z., Zhao, W.X., Wen, J.-R.: Chatcot: Toolaugmented chain-of-thought reasoning on chat-based large language models. arXiv preprint arXiv:2305.14323 (2023)

Cheng, Y., Zhang, C., Zhang, Z., Meng, X., Hong, S., Li, W., Wang, Z., Wang, Z.,

Yin, F., Zhao, J., et al.: Exploring large language model based intelligent agents: Definitions, methods, and prospects. arXiv preprint arXiv:2401.03428 (2024)

Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (long and Short Papers), pp. 4171-4186 (2019)

Drouin, A., Gasse, M., Caccia, M., Laradji, I.H., Del Verme, M., Marty, T., Boisvert, L., Thakkar, M., Cappart, Q., Vazquez, D., et al.: Workarena: How capable are web agents at solving common knowledge work tasks? arXiv preprint arXiv:2403.07718 (2024)

Du, Y., Li, S., Torralba, A., Tenenbaum, J.B., Mordatch, I.: Improving factuality and reasoning in language models through multiagent debate. In: Forty-first International Conference on Machine Learning (2023)

Dubois, Y., Li, C.X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., Guestrin, C., Liang, P.S., Hashimoto, T.B.: Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems 36, 30039-30069 (2023)

Dziri, N., Madotto, A., Za¨ıane, O., Bose, A.J.: Neural path hunter: Reducing hallucination in dialogue systems via path grounding. arXiv preprint arXiv:2104.08455 (2021)

Ding, H., Tao, S., Pang, L., Wei, Z., Gao, J., Ding, B., Shen, H., Chen, X.: Toolcoder: A systematic code-empowered tool learning framework for large language models. arXiv preprint arXiv:2502.11404 (2025)

El-Kishky, A., Wei, A., Saraiva, A., Minaiev, B., Selsam, D., Dohan, D., Song, F.,

Lightman, H., Clavera, I., Pachocki, J., Tworek, J., Kuhn, L., Kaiser, L., Chen, M., Schwarzer, M., Rohaninejad, M., McAleese, N., M¨urk, O., Garg, R., Shu, R., Sidor, S., Kosaraju, V., Zhou, W., contributors: Competitive programming with large reasoning models. arXiv preprint arXiv:2502.06807 (2025) European Union: EU AI Act. https://artificialintelligenceact.eu/. Accessed: 2025-1502 (2023)

Ford, M.: Architects of Intelligence: The Truth About AI from the People Building It. Packt Publishing Ltd, ??? (2018)

French, R.M.: Catastrophic forgetting in connectionist networks. Trends in cognitive sciences 3(4), 128-135 (1999)

Guo, T., Chen, X., Wang, Y., Chang, R., Pei, S., Chawla, N.V., Wiest, O., Zhang,

X.: Large language model based multi-agents: a survey of progress and challenges. In: Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pp. 8048-8057 (2024)

Ge, Z., Huang, H., Zhou, M., Li, J., Wang, G., Tang, S., Zhuang, Y.: Worldgpt: Empowering llm as multimodal world model. In: Proceedings of the 32nd ACM International Conference on Multimedia, pp. 7346-7355 (2024)

Gandhi, K., Lee, D., Grand, G., Liu, M., Cheng, W., Sharma, A., Goodman,

N.D.: Stream of search (sos): Learning to search in language. arXiv preprint arXiv:2404.03683 (2024)

Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., Neubig, G.: Pal: Program-aided language models. In: International Conference on Machine Learning, pp. 10764-10799 (2023). PMLR

Gridach, M., Nanavati, J., Abidine, K.Z.E., Mendes, L., Mack, C.: Agentic ai for scientific discovery: A survey of progress, challenges, and future directions. arXiv preprint arXiv:2503.08979 (2025)

Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)

Gao, Y., Sheng, T., Xiang, Y., Xiong, Y., Wang, H., Zhang, J.: Chat-rec: Towards interactive and explainable llms-augmented recommender system. arXiv preprint arXiv:2303.14524 (2023)

Gao, P., Xie, A., Mao, S., Wu, W., Xia, Y., Mi, H., Wei, F.: Meta reasoning for large language models. arXiv preprint arXiv:2406.11698 (2024)

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S.,

Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)

Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C.,

Puranik, S., He, H., Song, D., et al.: Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938 (2021)

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, D., Song, D.,

Steinhardt, J., Zou, J.: Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 (2021)

Huang, J., Chen, X., Mishra, S., Zheng, H.S., Yu, A.W., Song, X., Zhou, D.: Large language models cannot self-correct reasoning yet. In: The Twelfth International Conference on Learning Representations (2024)

Hazra, R., Dos Martires, P.Z., De Raedt, L.: Saycanpay: Heuristic planning with large language models using learnable domain knowledge. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 20123-20133 (2024)

Heath, A.: AI Experts Issue 22-word Warning About Extinction Risk. Accessed: 2025-04-16. https://www.theverge.com/2023/5/30/23742005/ ai-risk-warning-22-word-statement-google-deepmind-openai

Hua, W., Fan, L., Li, L., Mei, K., Ji, J., Ge, Y., Hemphill, L., Zhang, Y.: War and peace (waragent): Large language model-based multi-agent simulation of world wars. arXiv preprint arXiv:2311.17227 (2023)

Hao, S., Gu, Y., Ma, H., Hong, J.J., Wang, Z., Wang, D.Z., Hu, Z.: Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992 (2023)

Hao, R., Hu, L., Qi, W., Wu, Q., Zhang, Y., Nie, L.: Chatllm network: More brains, more intelligence. AI Open (2025)

Hu, S., Lu, C., Clune, J.: Automated design of agentic systems. arXiv preprint

Hosseini, A., Yuan, X., Malkin, N., Courville, A., Sordoni, A., Agarwal, R.: V-star: Training verifiers for self-taught reasoners. arXiv preprint arXiv:2402.06457 (2024)

Hong, S., Zheng, X., Chen, J., Cheng, Y., Wang, J., Zhang, C., Wang, Z., Yau, S.K.S.,

Lin, Z., Zhou, L., et al.: Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352 3(4), 6 (2023)

Hou, G., Zhang, W., Shen, Y., Tan, Z., Shen, S., Lu, W.: Entering real social world! benchmarking the theory of mind and socialization capabilities of llms from a firstperson perspective. arXiv preprint arXiv:2410.06195 (2024)

Imani, S., Du, L., Shrivastava, H.: Mathprompter: Mathematical reasoning using large language models. arXiv preprint arXiv:2303.05398 (2023)

Ifargan, T., Hafner, L., Kern, M., Alcalay, O., Kishony, R.: Autonomous llm-driven research--from data to human-verifiable research papers. NEJM AI 2(1), 2400555 (2025)

Ivison, H., Wang, Y., Liu, J., Wu, Z., Pyatkin, V., Lambert, N., Smith, N.A., Choi, Y., Hajishirzi, H.: Unpacking dpo and ppo: Disentangling best practices for learning from preference feedback. Advances in neural information processing systems 37, 36602-36633 (2024)

Jin, Y., Li, Z., Zhang, C., Cao, T., Gao, Y., Jayarao, P., Li, M., Liu, X., Sarkhel, R.,

Tang, X., et al.: Shopping mmlu: A massive multi-task online shopping benchmark for large language models. arXiv preprint arXiv:2410.20745 (2024)

Jang, J.Y., Shin, S., Gweon, G.: Minimal yet big impact: How ai agent back-channeling enhances conversational engagement through conversation persistence and context richness. In: Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 14509-14521 (2024)

Ju, T., Wang, Y., Ma, X., Cheng, P., Zhao, H., Wang, Y., Liu, L., Xie, J., Zhang, Z., Liu, G.: Flooding spread of manipulated knowledge in llm-based multi-agent communities. arXiv preprint arXiv:2407.07791 (2024)

Kojima, T., Gu, S.S., Reid, M., Matsuo, Y., Iwasawa, Y.: Large language models are zero-shot reasoners. Advances in neural information processing systems 35, 2219922213 (2022)

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., Stoica, I.: Efficient memory management for large language model serving with pagedattention. In: Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles (2023)

Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D.: Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 (2020)

Karpukhin, V., Oguz, B., Min, S., Lewis, P.S., Wu, L., Edunov, S., Chen, D., Yih,

W.-t.: Dense passage retrieval for open-domain question answering. In: EMNLP (1), pp. 6769-6781 (2020)

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A.,

Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al.: Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences 114(13), 3521-3526 (2017)

Kurzweil, R.: The Singularity Is Near: When Humans Transcend Biology. Viking Penguin, New York (2005)

Longo, L., Brcic, M., Cabitza, F., Choi, J., Confalonieri, R., Del Ser, J., Guidotti, R., Hayashi, Y., Herrera, F., Holzinger, A., et al.: Explainable artificial intelligence (xai) 2.0: A manifesto of open challenges and interdisciplinary research directions. Information Fusion 106, 102301 (2024)

Li, D., Cao, S., Griggs, T., Liu, S., Mo, X., Tang, E., Hegde, S., Hakhamaneshi, K.,

Patil, S.G., Zaharia, M., et al.: Llms can easily learn to reason from demonstrations structure, not content, is what matters! arXiv preprint arXiv:2502.07374 (2025)

Lin, B.Y., Fu, Y., Yang, K., Brahman, F., Huang, S., Bhagavatula, C., Ammanabrolu, P., Choi, Y., Ren, X.: Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks. Advances in Neural Information Processing Systems 36, 23813-23825 (2023)

Li, N., Gao, C., Li, M., Li, Y., Liao, Q.: Econagent: Large language model-empowered agents for simulating macroeconomic activities. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (2024)

Li, G., Hammoud, H., Itani, H., Khizbullin, D., Ghanem, B.: Camel: Communicative agents for" mind" exploration of large language model society. Advances in Neural Information Processing Systems 36, 51991-52008 (2023)

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Shi, S., Tu,

Z.: Encouraging divergent thinking in large language models through multi-agent debate. arXiv preprint arXiv:2305.19118 (2023)

Li, J., Hui, B., Qu, G., Yang, J., Li, B., Li, B., Wang, B., Qin, B., Geng, R., Huo, N., et al.: Can llm already serve as a database interface? a big bench for largescale database grounded text-to-sqls. Advances in Neural Information Processing Systems 36, 42330-42357 (2023)

Liu, B., Jiang, Y., Zhang, X., Liu, Q., Zhang, S., Biswas, J., Stone, P.: Llm+ p: Empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477 (2023)

Liu, S., Lu, Y., Chen, S., Hu, X., Zhao, J., Lu, Y., Zhao, Y.: Drugagent: Automating ai-aided drug discovery programming through llm multi-agent collaboration. arXiv preprint arXiv:2411.15692 (2024)

Lu, C., Lu, C., Lange, R.T., Foerster, J., Clune, J., Ha, D.: The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292 (2024)

Li, Z., Li, C., Zhang, M., Mei, Q., Bendersky, M.: Retrieval augmented generation or long-context llms? a comprehensive study and hybrid approach. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 881-893 (2024)

Lee, S., Park, S.H., Kim, S., Seo, M.: Aligning to thousands of preferences via system message generalization. Advances in Neural Information Processing Systems 37, 73783-73829 (2024)

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨uttler, H.,

Lewis, M., Yih, W.-t., Rockt¨aschel, T., et al.: Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33, 9459-9474 (2020)

Liu, H., Sferrazza, C., Abbeel, P.: Chain of hindsight aligns language models with feedback. arXiv preprint arXiv:2302.02676 (2023)

Lingam, V., Tehrani, B.O., Sanghavi, S., Gupta, G., Ghosh, S., Liu, L., Huan, J.,

Deoras, A.: Enhancing language model agents using diversity of thoughts. In: The Thirteenth International Conference on Learning Representations (2025)

Li, X., Wang, S., Zeng, S., Wu, Y., Yang, Y.: A survey on llm-based multi-agent systems: Workflow, infrastructure, and challenges. Vicinagearth 1(1), 9 (2024)

Liu, L., Zhang, C., Wu, L., Zhao, C., Hu, Z., He, M., Fan, J.: Instruct-of-reflection: Enhancing large language models iterative reflection capabilities via dynamic-meta instruction. arXiv preprint arXiv:2503.00902 (2025)

Lin, J., Zhao, H., Zhang, A., Wu, Y., Ping, H., Chen, Q.: Agentsims: An open-source sandbox for large language model evaluation. arXiv preprint arXiv:2308.04026 (2023)

Marr, B.: Generative AI Vs. Agentic AI: The Key Differences Everyone Needs to Know. Accessed: 2025-04-23. https://www.forbes.com/sites/bernardmarr/2025/02/ 03/generative-ai-vs-agentic-ai-the-key-differences-everyone-needs-to-know/

Miller, G.A.: The magical number seven, plus or minus two: Some limits on our capacity for processing information. Psychological Review 63(2), 81-97 (1956) https://doi.org/10.1037/h0043158

Maynez, J., Narayan, S., Bohnet, B., McDonald, R.: On faithfulness and factuality in abstractive summarization. arXiv preprint arXiv:2005.00661 (2020)

Mok, A.: Data is fueling the ai revolution. what happens when it runs out? California Magazine (2025)

Mondorf, P., Plank, B.: Beyond accuracy: Evaluating the reasoning behavior of large language models - a survey. arXiv preprint arXiv:2404.01869 (2024)

Manduchi, L., Pandey, K., Meister, C., Bamler, R., Cotterell, R., D¨aubener, S., Fellenz, S., Fischer, A., G¨artner, T., Kirchler, M., et al.: On the challenges and opportunities in generative ai. arXiv preprint arXiv:2403.00025 (2024)

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U.,

Dziri, N., Prabhumoye, S., Yang, Y., et al.: Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems 36, 46534-46594 (2023)

Mao, J., Ye, J., Qian, Y., Pavone, M., Wang, Y.: A language agent for autonomous driving. arXiv preprint arXiv:2311.10813 (2023)

Nottingham, K., Ammanabrolu, P., Suhr, A., Choi, Y., Hajishirzi, H., Singh, S., Fox,

R.: Do embodied agents dream of pixelated sheep: Embodied decision making using language guided world modelling. In: International Conference on Machine Learning, pp. 26311-26325 (2023). PMLR

Nolan, B.: Thousands of AI Researchers Were Asked If Tech Could Make Humans Extinct. A Surprising Number Said Yes. Accessed: 2025-04-16. https://www. businessinsider.com/ai-researchers-chance-tech-making-humans-extinct-2024-1

O'Grady, C.G., OG, C.: Agentic workflows in the practice of law--ai agents as ethics counsel. Arizona Legal Studies Discussion Paper, 25-03 (2024)

OpenAI: ChatGPT: Optimizing Language Models for Dialogue. https://openai.com/ blog/chatgpt/. Accessed: 2025-04-15 (2022)

OpenAI: Introducing the GPT Store. https://openai.com/blog/ introducing-the-gpt-store. Accessed: 2024-15-02 (2023)

OpenAI: Introducing GPT-4.5. Accessed: 2025-04-24. https://openai.com/index/ introducing-gpt-4-5

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C.,

Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35, 27730-27744 (2022) Oxford Languages: Definition of reasoning. https://languages.oup.com/. Accessed: 2025-04-15 (n.d.)

Paperswithcode: SOTA: Multi-task Language Understanding (MMLU). https:// paperswithcode.com/sota/multi-task-language-understanding-on-mmlu. Accessed: 2025-04-22 (2025)

Packer, C., Fang, V., Patil, S., Lin, K., Wooders, S., Gonzalez, J.: Memgpt: Towards llms as operating systems. (2023)

Pan, Y., Kong, D., Zhou, S., Cui, C., Leng, Y., Jiang, B., Liu, H., Shang, Y., Zhou, S., Wu, T., Wu, Z.: Webcanvas: Benchmarking web agents in online environments. arXiv preprint arXiv:2406.12373 (2024)

Piao, J., Lu, Z., Gao, C., Xu, F., Santos, F.P., Li, Y., Evans, J.: Emergence of human-like polarization among large language model agents. arXiv preprint arXiv:2501.05171 (2025)

Pallagani, V., Muppasani, B.C., Roy, K., Fabiano, F., Loreggia, A., Murugesan, K.,

Srivastava, B., Rossi, F., Horesh, L., Sheth, A.: On the prospects of incorporating large language models (llms) in automated planning and scheduling (aps). In: Proceedings of the International Conference on Automated Planning and Scheduling, vol. 34, pp. 432-444 (2024)

Park, J.S., O'Brien, J., Cai, C.J., Morris, M.R., Liang, P., Bernstein, M.S.: Generative agents: Interactive simulacra of human behavior. In: Proceedings of the 36th Annual Acm Symposium on User Interface Software and Technology, pp. 1-22 (2023)

Pounds, E.: What Is Agentic AI? Accessed: 2025-04-20. https://blogs.nvidia.com/ blog/what-is-agentic-ai/

Plaat, A., Duijn, M., Stein, N., Preuss, M., Putten, P., Batenburg, K.J.: Agentic large language models, a survey. arXiv preprint arXiv:2503.23037 (2025)

Paul, A., Yu, C.L., Susanto, E.A., Lau, N.W.L., Meadows, G.I.: Agentpeertalk: Empowering students through agentic-ai-driven discernment of bullying and joking in peer interactions in schools. arXiv preprint arXiv:2408.01459 (2024)

Parisi, A., Zhao, Y., Fiedel, N.: Talm: Tool augmented language models. arXiv preprint

Parisi, A., Zhao, Y., Fiedel, N.: Talm: Tool augmented language models. arXiv preprint

Patil, S.G., Zhang, T., Wang, X., Gonzalez, J.E.: Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems 37, 126544-126565 (2024)

Qi, X., Huang, K., Panda, A., Henderson, P., Wang, M., Mittal, P.: Visual adversarial examples jailbreak aligned large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 21527-21536 (2024)

Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., et al.: Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789 (2023)

Qiao, B., Li, L., Zhang, X., He, S., Kang, Y., Zhang, C., Yang, F., Dong, H., Zhang, J., Wang, L., Ma, M., Zhao, P., Qin, S., Qin, X., Du, C., Xu, Y., Lin, Q., Rajmohan, S., Zhang, D.: Taskweaver: A code-first agent framework. arXiv preprint arXiv:2311.17541 (2023)

Ruan, J., Chen, Y., Zhang, B., Xu, Z., Bao, T., Mao, H., Li, Z., Zeng, X., Zhao, R., et al.: Tptu: Task planning and tool usage of large language model-based ai agents. In: NeurIPS 2023 Foundation Models for Decision Making Workshop (2023)

Razghandi, A., Hosseini, S.M.H., Baghshah, M.S.: Cer: Confidence enhanced reasoning in llms. arXiv preprint arXiv:2502.14634 (2025)

Richards, T.B.: AutoGPT: An autonomous GPT-4 experiment. https://github.com/ Torantulino/Auto-GPT. GitHub repository (2023)

Russell, S., Norvig, P.: Artificial Intelligence: A Modern Approach, 4th edn. Pearson, ??? (2021)

Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems 36, 53728-53741 (2023)

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.: Language models are unsupervised multitask learners. OpenAI blog 1(8), 9 (2019)

Reed, S., Zolna, K., Parisotto, E., Colmenarejo, S.G., Novikov, A., Barth-Maron, G.,

Gimenez, M., Sulsky, Y., Kay, J., Springenberg, J.T., et al.: A generalist agent. arXiv preprint arXiv:2205.06175 (2022)

Shavit, Y., Agarwal, S., Brundage, M., Adler, S., O'Keefe, C., Campbell, R.,

Lee, T., Mishkin, P., Eloundou, T., Hickey, A., Slama, K., Ahmad, L., McMillan, P., Vallone, A., Passos, A., Robinson, D.G.: Practices for Governing Agentic AI Systems. Accessed: 2025-04-23. https://openai.com/index/ practices-for-governing-agentic-ai-systems/

Schultz, J., Adamek, J., Jusup, M., Lanctot, M., Kaisers, M., Perrin, S., Hennes, D.,

Shar, J., Lewis, C., Ruoss, A., et al.: Mastering board games by external and internal planning with language models. arXiv preprint arXiv:2412.12119 (2024)

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S.: Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems 36, 8634-8652 (2023)

Schneider, J.: Explainable generative ai (genxai): A survey, conceptualization, and research agenda. Artificial Intelligence Review 57(11), 289 (2024)

Schneider, J.: What comes after transformers?-a selective survey connecting ideas in deep learning. arXiv preprint arXiv:2408.00386 (2024)

Schneider, J.: Improving next tokens via second-last predictions with generate and refine. (2025)

Schick, T., Dwivedi-Yu, J., Dess`ı, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., Scialom, T.: Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems 36, 68539-68551 (2023)

Singh, A., Ehtesham, A., Kumar, S., Khoei, T.T.: Agentic retrieval-augmented generation: A survey on agentic rag. arXiv preprint arXiv:2501.09136 (2025)

Schneider, J., Haag, S., Kruse, L.C.: Negotiating with llms: Prompt hacks, skill gaps, and reasoning deficits. arXiv preprint arXiv:2312.03720 (2023)

Shen, S., Huang, F., Zhao, Z., Liu, C., Zheng, T., Zhu, D.: Long is more important than difficult for training reasoning models. arXiv preprint arXiv:2503.18069 (2025)

Singh, M.P.: Consent as a foundation for responsible autonomy. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, pp. 12301-12306 (2022)

Schneider, J., Meske, C., Kuss, P.: Foundation models: A new paradigm for artificial intelligence. Business & Information Systems Engineering 66, 221-231 (2024)

Schneider, J., Prabhushankar, M.: Understanding and leveraging the learning phases of neural networks. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 14886-14893 (2024)

Shen, Y., Song, K., Tan, X., Li, D., Lu, W., Zhuang, Y.: Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems 36, 38154-38180 (2023)

Sudarshan, M., Shih, S., Yee, E., Yang, A., Zou, J., Chen, C., Zhou, Q., Chen, L.,

Singhal, C., Shih, G.: Agentic llm workflows for generating patient-friendly medical reports. arXiv preprint arXiv:2408.01112 (2024)

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y.,

Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)

Sprague, Z., Yin, F., Rodriguez, J.D., Jiang, D., Wadhwa, M., Singhal, P., Zhao, X.,

Ye, X., Mahowald, K., Durrett, G.: To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. arXiv preprint arXiv:2409.12183 (2024)

Team, G., Georgiev, P., Lei, V.I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al.: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024)

Touvron, H., Lavril, T., Martinet, X., Others: The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024)

Turpin, M., Michael, J., Perez, E., Bowman, S.: Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. Advances in Neural Information Processing Systems 36, 74952-74965 (2023)

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023)

Walke, H.R., Black, K., Zhao, T.Z., Vuong, Q., Zheng, C., Hansen-Estruch, P., He, A.W., Myers, V., Kim, M.J., Du, M., et al.: Bridgedata v2: A dataset for robot learning at scale. In: Conference on Robot Learning, pp. 1723-1736 (2023). PMLR

Wang, Z., Cai, S., Chen, G., Liu, A., Ma, X., Liang, Y.: Describe, explain, plan and select: Interactive planning with large language models enables open-world multitask agents. arXiv preprint arXiv:2302.01560 (2023)

Wang, X., Chen, Y., Yuan, L., Zhang, Y., Li, Y., Peng, H., Ji, H.: Executable code actions elicit better llm agents. In: Forty-first International Conference on Machine Learning (2024)

White, J., Fu, Q., Hays, S., Sandborn, M., Olea, C., Gilbert, H., Elnashar, A., Spencer-

Smith, J., Schmidt, D.C.: A prompt pattern catalog to enhance prompt engineering with chatgpt. arXiv preprint arXiv:2302.11382 (2023) W¨olflein, G., Ferber, D., Truhn, D., Arandjelovi´c, O., Kather, J.N.: Llm agents making agent tools. arXiv preprint arXiv:2502.11705 (2025)

Wu, J., Feng, M., Zhang, S., Che, F., Wen, Z., Tao, J.: Beyond examples: Highlevel automated reasoning paradigm in in-context learning via mcts. arXiv preprint arXiv:2411.18478 (2024)

Williams, R., Hosseinichimeh, N., Majumdar, A., Ghaffarzadegan, N.: Epidemic modeling with generative agents. arXiv preprint arXiv:2307.04986 (2023)

Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., et al.: A survey on large language model based autonomous agents. Frontiers of Computer Science 18(6), 186345 (2024)

Wang, Z., Mao, S., Wu, W., Ge, T., Wei, F., Ji, H.: Unleashing the emergent cognitive synergy in large language models: A task-solving agent through multi-persona selfcollaboration. arXiv preprint arXiv:2307.05300 (2023)

Wang, W., Ma, Z., Wang, Z., Wu, C., Chen, W., Li, X., Yuan, Y.: A survey of llm-based agents in medicine: How far are we from baymax? arXiv preprint arXiv:2502.11211 (2025)

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al.: Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In: The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2024)

Wohlin, C.: Guidelines for snowballing in systematic literature studies and a replication in software engineering. In: Proceedings of the 18th International Conference on Evaluation and Assessment in Software Engineering, pp. 1-10 (2014)

Wang, Z.M., Peng, Z., Que, H., Liu, J., Zhou, W., Wu, Y., Guo, H., Gan, R., Ni, Z., Yang, J., et al.: Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models. arXiv preprint arXiv:2310.00746 (2023)

Wang, J., Shi, E., Hu, H., Ma, C., Liu, Y., Wang, X., Yao, Y., Liu, X., Ge, B., Zhang,

S.: Large language models for robotics: Opportunities, challenges, and perspectives. Journal of Automation and Intelligence (2024)

Wu, Y., Tang, X., Mitchell, T.M., Li, Y.: Smartplay: A benchmark for llms as intelligent agents. arXiv preprint arXiv:2310.01557 (2023)

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou,

D.: Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022)

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 24824-24837 (2022)

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35, 24824-24837 (2022)

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., Anandkumar,

A.: Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 (2023)

Wang, X., Zhou, D.: Chain-of-thought reasoning without prompting. arXiv preprint

Wang, L., Zhang, J., Chen, X., Lin, Y., Song, R., Zhao, W.X., Wen, J.-R.: Recagent: A novel simulation paradigm for recommender systems. arXiv preprint arXiv:2306.02552 (2023)

Wang, Y., Zhao, S., Wang, Z., Huang, H., Fan, M., Zhang, Y., Wang, Z., Wang, H., Liu, T.: Strategic chain-of-thought: Guiding accurate reasoning in llms through strategy elicitation. arXiv preprint arXiv:2409.03271 (2024)

Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang, J., Jin, S., Zhou, E., et al.: The rise and potential of large language model based agents: A survey. Science China Information Sciences 68(2), 121101 (2025)

Xiao, Y., Sun, E., Luo, D., Wang, W.: Tradingagents: Multi-agents llm financial trading framework. arXiv preprint arXiv:2412.20138 (2024)

Xiang, J., Tao, T., Gu, Y., Shu, T., Wang, Z., Yang, Z., Hu, Z.: Language models meet world models: Embodied experiences enhance language models. Advances in neural information processing systems 36, 75392-75412 (2023)

Xu, S., Xie, W., Zhao, L., He, P.: Chain of draft: Thinking faster by writing less. arXiv preprint arXiv:2502.18600 (2025) arXiv:2502.18600 [cs.CL]

Yasunaga, M., Chen, X., Li, Y., Pasupat, P., Leskovec, J., Liang, P., Chi, E.H., Zhou,

D.: Large language models as analogical reasoners. arXiv preprint arXiv:2310.01714 (2023)

Yuan, L., Chen, Y., Wang, X., Fung, Y.R., Peng, H., Ji, H.: Craft: Customizing llms by creating and retrieving from specialized toolsets. arXiv preprint arXiv:2309.17428 (2023)

Yehudai, A., Eden, L., Li, A., Uziel, G., Zhao, Y., Bar-Haim, R., Cohan, A.,

Shmueli-Scheuer, M.: Survey on evaluation of llm-based agents. arXiv preprint arXiv:2503.16416 (2025)

Yan, S.-Q., Gu, J.-C., Zhu, Y., Ling, Z.-H.: Corrective retrieval augmented generation. arXiv preprint arXiv:2401.15884 (2024)

Yamamoto, A., Ito, O., Katagiri, A., Koike, Y.: Dynamic knowledge integration in multi-agent systems for content inference. In: ICLR 2025 Workshop on Agentic AI for Science (2025)

Yang, J., Jimenez, C., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., Press, O.: Sweagent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems 37, 50528-50652 (2024)

Yuan, S., Song, K., Chen, J., Tan, X., Li, D., Yang, D.: Evoagent: Towards automatic multi-agent generation via evolutionary algorithms. arXiv preprint arXiv:2406.14228 (2024)

Yang, L., Yu, Z., Cui, B., Wang, M.: Reasonflux: Hierarchical llm reasoning via scaling thought templates. arXiv preprint arXiv:2502.06772 (2025)

Yu, Y., Yao, Z., Li, H., Deng, Z., Jiang, Y., Cao, Y., Chen, Z., Suchow, J., Cui, Z., Liu, R., et al.: Fincon: A synthesized llm multi-agent system with conceptual verbal reinforcement for enhanced financial decision making. Advances in Neural Information Processing Systems 37, 137010-137045 (2024)

Yu, W., Yang, Z., Wan, J., Song, S., Tang, J., Cheng, W., Liu, Y., Bai, X.: Omniparser v2: Structured-points-of-thought for unified visual text parsing and its generality to multimodal large language models. arXiv preprint arXiv:2502.16161 (2025)

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., Narasimhan, K.: Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems 36, 11809-11822 (2023)

Yang, L., Yu, Z., Zhang, T., Cao, S., Xu, M., Zhang, W., Gonzalez, J.E., Cui, B.: Buffer of thoughts: Thought-augmented reasoning with large language models. Advances in Neural Information Processing Systems 37, 113519-113544 (2024)

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y.: React: Synergizing reasoning and acting in language models. In: International Conference on Learning Representations (ICLR) (2023)

Zhang, Q., Chen, S., Bei, Y., Yuan, Z., Zhou, H., Hong, Z., Dong, J., Chen, H., Chang, Y., Huang, X.: A survey of graph retrieval-augmented generation for customized large language models. arXiv preprint arXiv:2501.13958 (2025)

Zhang, H., Chen, J., Jiang, F., Yu, F., Chen, Z., Li, J., Chen, G., Wu, X., Zhang, Z.,

Xiao, Q., et al.: Huatuogpt, towards taming language model to be a doctor. arXiv preprint arXiv:2305.15075 (2023)

Zhang, H., Du, W., Shan, J., Zhou, Q., Du, Y., Tenenbaum, J.B., Shu, T., Gan, C.: Building cooperative embodied agents modularly with large language models. arXiv preprint arXiv:2307.02485 (2023)

Zhang, J., Hou, Y., Xie, R., Sun, W., McAuley, J., Zhao, W.X., Lin, L., Wen, J.-R.:

Agentcf: Collaborative learning with autonomous language agents for recommender systems. In: Proceedings of the ACM Web Conference 2024, pp. 3679-3689 (2024)

Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., Huang, G.: Expel: Llm agents are experiential learners. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 19632-19642 (2024)

Zhong, T., Liu, Z., Pan, Y., Zhang, Y., Zhou, Y., Liang, S., Wu, Z., Lyu, Y., Shu, P.,

Yu, X., et al.: Evaluation of openai o1: Opportunities and challenges of agi. arXiv preprint arXiv:2409.18486 (2024)

Zheng, C., Liu, Z., Xie, E., Li, Z., Li, Y.: Progressive-hint prompting improves reasoning in large language models. arXiv preprint arXiv:2304.09797 (2023) Zoe Kleinmann: Why Google's 'woke' AI Problem Won't Be an Easy Fix. Accessed: 2025-04-23. https://www.bbc.com/news/technology-68412620

Zhang, T., Patil, S.G., Jain, N., Shen, S., Zaharia, M., Stoica, I., Gonzalez, J.E.: Raft: Adapting language model to domain specific rag. In: First Conference on Language Modeling (2024)

Zhou, P., Pujara, J., Ren, X., Chen, X., Cheng, H.-T., Le, Q.V., Chi, E., Zhou, D.,

Mishra, S., Zheng, H.S.: Self-discover: Large language models self-compose reasoning structures. Advances in Neural Information Processing Systems 37, 126032-126058 (2024)

Zhang, Y., Sun, R., Chen, Y., Pfister, T., Zhang, R., Arik, S.: Chain of agents: Large language models collaborating on long-context tasks. Advances in Neural Information Processing Systems 37, 132208-132237 (2024)

Zhou, D., Sch¨arli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C.,

Bousquet, O., Le, Q., et al.: Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625 (2022)

Ziegler, D.M., Stiennon, N., Wu, J., Brown, T.B., Radford, A., Amodei, D., Christiano, P., Irving, G.: Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593 (2019)

Zhang, W., Shen, Y., Wu, L., Peng, Q., Wang, J., Zhuang, Y., Lu, W.: Self-contrast: Better reflection through inconsistent solving perspectives. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3602-3622 (2024)

Zelikman, E., Wu, Y., Mu, J., Goodman, N.: Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems 35, 15476-15488 (2022)

Zheng, L., Wang, R., Wang, X., An, B.: Synapse: Trajectory-as-exemplar prompting with memory for computer control. arXiv preprint arXiv:2306.07863 (2023)

Zhou, S., Xu, F.F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y.,

Fried, D., et al.: Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854 (2023)

Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., Wang, Y.-X.: Language agent tree search unifies reasoning acting and planning in language models. arXiv preprint arXiv:2310.04406 (2023)

Zhang, D., Zhoubian, S., Hu, Z., Yue, Y., Dong, Y., Tang, J.: Rest-mcts*: Llm selftraining via process reward guided tree search. Advances in Neural Information Processing Systems 37, 64735-64772 (2024)

Zhang, C., Zhang, C., Zheng, S., Qiao, Y., Li, C., Zhang, M., Dam, S.K., Thwal, C.M.,

Tun, Y.L., Huy, L.L., et al.: A complete survey on generative ai (aigc): Is chatgpt from gpt-4 to gpt-5 all you need? arXiv preprint arXiv:2303.11717 (2023)
