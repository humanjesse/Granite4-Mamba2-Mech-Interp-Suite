"""SAE training: collect activations from model residual stream, train sparse autoencoder.

Pipeline:
1. collect_activations() — hook a layer's output, run diverse text, stack hidden states
2. train_sae() — train the SAE on collected activations with MSE + L1 loss
"""

import torch
from dataclasses import dataclass, field

from .sparse_autoencoder import SparseAutoencoder


@dataclass
class SAETrainingConfig:
    """Configuration for SAE training."""

    layer_idx: int = 16
    expansion_factor: int = 8
    l1_coefficient: float = 1e-3
    learning_rate: float = 3e-4
    num_activations: int = 50_000
    batch_size: int = 256
    num_epochs: int = 1
    normalize_decoder_every: int = 100


@dataclass
class SAETrainingResult:
    """Result of SAE training."""

    sae: SparseAutoencoder
    config: SAETrainingConfig
    final_loss: float
    final_reconstruction_loss: float
    final_l1_loss: float
    loss_history: list[float] = field(default_factory=list)
    reconstruction_history: list[float] = field(default_factory=list)
    l1_history: list[float] = field(default_factory=list)
    mean_active_features: float = 0.0
    dead_features: int = 0
    num_latent_features: int = 0


def _get_builtin_texts() -> list[str]:
    """Diverse text corpus for activation collection.

    Covers science, nature, code, history, math, conversation, and
    importantly includes bee/insect content so the SAE can learn
    those features.
    """
    return [
        # Science & nature
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The human genome contains approximately three billion base pairs.",
        "Neurons communicate through electrical impulses and chemical synapses.",
        "Plate tectonics describes the movement of Earth's lithospheric plates.",
        "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        "DNA replication is a semiconservative process involving helicase and polymerase.",
        "Black holes form when massive stars collapse under their own gravity.",
        "The periodic table organizes elements by atomic number and properties.",
        "Mitochondria are the powerhouses of the cell, producing ATP through oxidative phosphorylation.",
        "Quantum entanglement allows particles to be correlated across vast distances.",
        "Volcanic eruptions release magma, ash, and gases from deep within the Earth.",
        "The water cycle involves evaporation, condensation, precipitation, and collection.",
        "Antibiotics work by targeting bacterial cell walls or protein synthesis.",
        "The theory of relativity fundamentally changed our understanding of space and time.",
        "Coral reefs support approximately 25 percent of all marine species.",
        "Gravitational waves were first detected by LIGO in 2015.",
        "Enzymes are biological catalysts that speed up chemical reactions.",
        "The electromagnetic spectrum ranges from radio waves to gamma rays.",
        "Glaciers form from accumulated snow that compresses into dense ice over centuries.",
        "Nuclear fusion powers the sun by combining hydrogen atoms into helium.",
        # Bee & insect content (important for feature learning)
        "Honeybees are essential pollinators responsible for one-third of our food supply.",
        "The queen bee is the only reproductive female in a honeybee colony.",
        "Worker bees perform a waggle dance to communicate the location of nectar sources.",
        "A single honeybee colony can contain up to 60,000 individual bees.",
        "Bees produce honey by collecting nectar and evaporating its water content.",
        "The hexagonal structure of honeycomb is a marvel of natural engineering efficiency.",
        "Colony collapse disorder has caused significant declines in bee populations worldwide.",
        "Bumblebees can fly at altitudes of up to 29,000 feet above sea level.",
        "Royal jelly is a protein-rich secretion that transforms larvae into queen bees.",
        "Bee venom contains melittin, a compound studied for its anti-inflammatory properties.",
        "Butterflies undergo complete metamorphosis from caterpillar to adult.",
        "Ants communicate primarily through chemical signals called pheromones.",
        "Dragonflies can fly backwards and hover like helicopters.",
        "Ladybugs are voracious predators of aphids and other garden pests.",
        "Fireflies produce bioluminescence through a chemical reaction in their abdomens.",
        "Beetles comprise the largest order of insects with over 400,000 known species.",
        "Mosquitoes are attracted to carbon dioxide and body heat when seeking hosts.",
        "Termites build elaborate mound structures with sophisticated ventilation systems.",
        "Praying mantises are ambush predators with powerful raptorial forelegs.",
        "Cicadas emerge from underground in synchronized broods after years of development.",
        # Technology & computing
        "Machine learning models learn patterns from data through gradient descent optimization.",
        "Transformer architectures use self-attention mechanisms to process sequential data.",
        "The internet protocol suite enables communication between billions of connected devices.",
        "Encryption algorithms protect sensitive data by converting it into unreadable ciphertext.",
        "Graphics processing units accelerate parallel computations in deep learning workloads.",
        "Version control systems track changes to code and enable collaborative development.",
        "Cloud computing provides on-demand access to shared pools of computing resources.",
        "Relational databases organize data into tables with rows and columns linked by keys.",
        "Containerization packages applications and dependencies into portable runtime environments.",
        "Natural language processing enables computers to understand and generate human language.",
        "Quantum computing uses qubits that can exist in superposition states.",
        "Operating systems manage hardware resources and provide interfaces for applications.",
        "The TCP handshake establishes reliable connections between networked devices.",
        "Garbage collection automatically reclaims memory occupied by unused objects.",
        "Microservices architecture decomposes applications into small independent services.",
        # History & culture
        "The invention of the printing press in 1440 revolutionized the spread of knowledge.",
        "Ancient Egyptian civilization flourished along the banks of the Nile River.",
        "The Industrial Revolution transformed manufacturing through mechanization and steam power.",
        "The Renaissance marked a cultural rebirth in Europe beginning in the 14th century.",
        "The Silk Road connected Eastern and Western civilizations through trade routes.",
        "Democracy originated in ancient Athens where citizens participated directly in governance.",
        "The space race between superpowers led to the first moon landing in 1969.",
        "The Great Wall of China stretches over 13,000 miles across northern China.",
        "The French Revolution of 1789 fundamentally altered the political landscape of Europe.",
        "Ancient Roman engineering innovations included aqueducts, roads, and concrete construction.",
        # Mathematics
        "The Fibonacci sequence appears in many natural patterns including spiral shells.",
        "Euler's identity connects five fundamental mathematical constants in one elegant equation.",
        "Calculus provides tools for analyzing rates of change and accumulation.",
        "Prime numbers have exactly two distinct positive divisors: one and themselves.",
        "Linear algebra studies vector spaces and linear transformations between them.",
        "The Pythagorean theorem relates the sides of a right triangle.",
        "Probability theory provides a mathematical framework for quantifying uncertainty.",
        "Topology studies properties preserved under continuous deformation of shapes.",
        "The central limit theorem explains why many distributions approximate the normal curve.",
        "Graph theory models relationships between objects using vertices and edges.",
        # Geography & weather
        "The Amazon rainforest produces approximately 20 percent of the world's oxygen.",
        "Ocean currents distribute heat around the globe affecting regional climates.",
        "Earthquakes occur when tectonic stress exceeds the strength of rock formations.",
        "The Sahara Desert is the largest hot desert spanning much of North Africa.",
        "Hurricanes form over warm tropical ocean waters and can produce devastating winds.",
        "The Mariana Trench is the deepest known point in Earth's oceans.",
        "Mountain ranges form where tectonic plates converge and push upward.",
        "The jet stream is a fast-flowing air current in the upper atmosphere.",
        "Tsunamis are generated by underwater earthquakes or volcanic eruptions.",
        "Permafrost contains vast amounts of frozen organic carbon in Arctic regions.",
        # Food & agriculture
        "Rice cultivation provides the staple food for more than half the world's population.",
        "Fermentation is a metabolic process that produces alcohol and carbon dioxide.",
        "Crop rotation helps maintain soil fertility and reduces pest buildup.",
        "Chocolate is made from roasted and ground cacao beans mixed with sugar.",
        "Irrigation systems deliver water to crops in regions with insufficient rainfall.",
        "Coffee beans are actually the seeds of berries from Coffea plants.",
        "Selective breeding has dramatically increased agricultural yields over centuries.",
        "Sourdough bread relies on wild yeast and bacteria for leavening.",
        "Hydroponics grows plants in nutrient-rich water solutions without soil.",
        "The Green Revolution introduced high-yield crop varieties to developing nations.",
        # Language & literature
        "Shakespeare wrote 37 plays and 154 sonnets that profoundly influenced English literature.",
        "The Rosetta Stone enabled scholars to decipher ancient Egyptian hieroglyphs.",
        "Mandarin Chinese is the most widely spoken native language in the world.",
        "Poetry uses rhythm, imagery, and figurative language to evoke emotional responses.",
        "The Epic of Gilgamesh is one of the earliest known works of literature.",
        "Linguistic diversity encompasses approximately 7,000 languages spoken worldwide.",
        "Grammar provides the structural rules that govern the composition of sentences.",
        "The invention of writing around 3400 BCE marked the beginning of recorded history.",
        "Metaphors and similes create vivid comparisons that enhance written expression.",
        "Translation requires understanding cultural context beyond literal word meanings.",
        # Medicine & health
        "Vaccines stimulate the immune system to produce antibodies against specific pathogens.",
        "The cardiovascular system circulates blood through arteries, veins, and capillaries.",
        "Stem cells have the remarkable ability to develop into many different cell types.",
        "Antibiotics target bacterial infections but are ineffective against viral diseases.",
        "MRI scanners use powerful magnetic fields to create detailed images of internal organs.",
        "The human brain contains approximately 86 billion neurons forming trillions of connections.",
        "Regular exercise strengthens the cardiovascular system and improves mental health.",
        "Gene therapy aims to treat diseases by modifying or replacing defective genes.",
        "The placebo effect demonstrates the powerful influence of expectation on healing.",
        "Sleep is essential for memory consolidation and cellular repair processes.",
        # Economics & business
        "Supply and demand determine market prices in competitive economic systems.",
        "Inflation occurs when the general price level of goods and services rises over time.",
        "Gross domestic product measures the total economic output of a country.",
        "Stock markets provide platforms for buying and selling ownership shares in companies.",
        "International trade allows countries to specialize in producing their comparative advantages.",
        "Central banks use monetary policy tools to manage inflation and employment.",
        "Entrepreneurship drives innovation by creating new products, services, and business models.",
        "Compound interest accelerates wealth growth by earning returns on previous returns.",
        "Fiscal policy involves government decisions about taxation and spending.",
        "Globalization has increased economic interconnectedness between nations worldwide.",
        # Philosophy & psychology
        "Cognitive biases are systematic patterns of deviation from rational judgment.",
        "The scientific method involves forming hypotheses and testing them through experiments.",
        "Existentialism emphasizes individual freedom and the responsibility of personal choice.",
        "Memory formation involves encoding, storage, and retrieval of information.",
        "Ethics examines questions of right and wrong conduct and moral principles.",
        "Pavlov's experiments demonstrated classical conditioning in animal behavior.",
        "Critical thinking involves analyzing arguments and evaluating evidence objectively.",
        "The unconscious mind influences behavior in ways not immediately accessible to awareness.",
        "Stoicism teaches the development of self-control and fortitude to overcome emotions.",
        "Neuroplasticity allows the brain to reorganize itself by forming new neural connections.",
        # Code snippets
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id",
        "const fetchData = async () => { const response = await fetch(url); return response.json(); }",
        "for i in range(len(matrix)): for j in range(len(matrix[0])): result[i][j] = matrix[i][j] * 2",
        "class Node: def __init__(self, val): self.val = val; self.left = None; self.right = None",
        "import numpy as np; data = np.random.randn(1000, 768); mean = data.mean(axis=0)",
        "git commit -m 'refactor authentication middleware to use JWT tokens'",
        "docker build -t myapp:latest . && docker run -p 8080:8080 myapp:latest",
        "CREATE TABLE measurements (id SERIAL PRIMARY KEY, value FLOAT, timestamp TIMESTAMPTZ)",
        "model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)",
        # Conversation / casual
        "The best way to learn a new skill is through consistent practice and feedback.",
        "Traveling to different countries broadens your perspective on life and culture.",
        "Reading books before bed helps reduce stress and improve sleep quality.",
        "Cooking at home is generally healthier and more economical than eating out.",
        "Learning a musical instrument develops coordination, patience, and creativity.",
        "Regular walking in nature has been shown to reduce anxiety and improve mood.",
        "Writing in a journal helps organize thoughts and process complex emotions.",
        "Volunteering in your community creates meaningful connections and a sense of purpose.",
        "Setting clear goals and breaking them into small steps increases the chance of success.",
        "Good communication involves active listening as much as clear expression.",
        # More diverse science
        "Photovoltaic cells convert sunlight directly into electrical energy using semiconductors.",
        "CRISPR technology enables precise editing of genetic sequences in living organisms.",
        "The James Webb Space Telescope captures infrared light from the earliest galaxies.",
        "Superconductors conduct electricity with zero resistance below critical temperatures.",
        "Carbon nanotubes exhibit extraordinary strength and unique electrical properties.",
        "Epigenetics studies how environmental factors can modify gene expression without altering DNA.",
        "The Standard Model describes fundamental particles and three of the four known forces.",
        "Oceanographic research reveals complex ecosystems thriving near hydrothermal vents.",
        "Biodegradable plastics decompose through natural biological processes.",
        "Artificial photosynthesis aims to replicate natural energy conversion processes.",
    ]


def collect_activations(
    model,
    tokenizer,
    layer_idx: int,
    num_activations: int,
    device: str = "cpu",
    dataset_texts: list[str] | None = None,
    max_seq_len: int = 128,
    progress_callback=None,
) -> torch.Tensor:
    """Collect residual stream activations by hooking a model layer.

    Runs text through the model, captures the hidden state output of the
    specified layer at every token position, and stacks them into a single
    tensor.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        layer_idx: Which layer to hook (0-indexed).
        num_activations: Target number of activation vectors to collect.
        device: Device for inference.
        dataset_texts: Text corpus. Defaults to built-in diverse set.
        max_seq_len: Maximum tokens per text.
        progress_callback: Called with (collected_so_far, total).

    Returns:
        [N, hidden_size] float32 tensor where N >= num_activations.
    """
    if dataset_texts is None:
        dataset_texts = _get_builtin_texts()

    all_activations = []
    collected = 0

    # Register hook on the target layer
    layer = model.model.layers[layer_idx]
    storage = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        storage["hidden"] = hidden.detach()

    hook = layer.register_forward_hook(hook_fn)

    try:
        text_idx = 0
        while collected < num_activations:
            text = dataset_texts[text_idx % len(dataset_texts)]
            text_idx += 1

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                model(input_ids=input_ids)

            hidden = storage["hidden"]  # [1, seq_len, hidden_size]
            # Take all token positions
            acts = hidden[0].float().cpu()  # [seq_len, hidden_size]
            all_activations.append(acts)
            collected += acts.shape[0]

            if progress_callback:
                progress_callback(min(collected, num_activations), num_activations)
    finally:
        hook.remove()

    # Stack and trim to exactly num_activations
    result = torch.cat(all_activations, dim=0)[:num_activations]
    return result


def train_sae(
    activations: torch.Tensor,
    config: SAETrainingConfig,
    progress_callback=None,
    device: str = "cpu",
) -> SAETrainingResult:
    """Train a sparse autoencoder on pre-collected activations.

    Loss = MSE(x, x_hat) + l1_coefficient * mean(|z|)

    Args:
        activations: [N, hidden_size] tensor of activation vectors in float32.
        config: Training configuration.
        progress_callback: Called with (step, total_steps).
        device: Device to train on ("cpu" or "cuda").

    Returns:
        SAETrainingResult with trained SAE and training metrics.
    """
    input_dim = activations.shape[1]
    latent_dim = input_dim * config.expansion_factor

    sae = SparseAutoencoder(input_dim, latent_dim).to(device)
    activations = activations.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)

    n_samples = activations.shape[0]
    total_steps = config.num_epochs * ((n_samples + config.batch_size - 1) // config.batch_size)
    step = 0

    loss_history = []
    reconstruction_history = []
    l1_history = []

    # Track which features ever activate
    feature_ever_active = torch.zeros(latent_dim, dtype=torch.bool, device=device)

    for epoch in range(config.num_epochs):
        # Shuffle each epoch
        perm = torch.randperm(n_samples, device=device)
        shuffled = activations[perm]

        for batch_start in range(0, n_samples, config.batch_size):
            batch = shuffled[batch_start : batch_start + config.batch_size]

            x_hat, z = sae(batch)
            recon_loss = torch.nn.functional.mse_loss(x_hat, batch)
            l1_loss = z.abs().mean()
            total_loss = recon_loss + config.l1_coefficient * l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Periodically normalize decoder
            step += 1
            if step % config.normalize_decoder_every == 0:
                sae._normalize_decoder()

            loss_history.append(total_loss.item())
            reconstruction_history.append(recon_loss.item())
            l1_history.append(l1_loss.item())

            # Track active features
            feature_ever_active |= (z.detach() > 1e-5).any(dim=0)

            if progress_callback:
                progress_callback(step, total_steps)

    # Final decoder normalization
    sae._normalize_decoder()

    # Compute stats
    sae.eval()
    with torch.no_grad():
        _, z_all = sae(activations[:min(10000, n_samples)])
        mean_active = (z_all > 0).float().sum(dim=1).mean().item()

    dead_features = int((~feature_ever_active).sum().item())

    # Move SAE back to CPU for portability
    sae = sae.cpu()

    return SAETrainingResult(
        sae=sae,
        config=config,
        final_loss=loss_history[-1] if loss_history else 0.0,
        final_reconstruction_loss=reconstruction_history[-1] if reconstruction_history else 0.0,
        final_l1_loss=l1_history[-1] if l1_history else 0.0,
        loss_history=loss_history,
        reconstruction_history=reconstruction_history,
        l1_history=l1_history,
        mean_active_features=mean_active,
        dead_features=dead_features,
        num_latent_features=latent_dim,
    )
