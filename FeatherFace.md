## 2. Benchmarks Performance Mobile

### 2.1 Tests multi-appareils et latence d'inférence

L'évaluation de performance mobile couvre un spectrum représentatif d'appareils disponibles en milieu scolaire, des smartphones haut de gamme aux dispositifs entrée de gamme. Les mesures incluent latence d'inférence, consommation mémoire, et stabilité thermique.

**Tableau 5.4** : Benchmarks latence d'inférence (milliseconds par image)

| Appareil | Processeur | FeatherFace V1 | FeatherFace V2 | Speedup | Objectif |
|----------|------------|----------------|----------------|---------|----------|
| iPhone 13 Pro | A15 Bionic | 45.2ms | 28.1ms | 1.61x | <50ms ✓ |
| iPhone SE 2022 | A15 Bionic | 52.8ms | 31.7ms | 1.67x | <50ms ✓ |
| Galaxy S22 | Snapdragon 8 Gen 1 | 67.3ms | 39.4ms | 1.71x | <80ms ✓ |
| Galaxy A52 | Snapdragon 720G | 142.1ms | 89.7ms | 1.58x | <120ms ✓ |
| Redmi Note 10 | Snapdragon 678 | 186.4ms | 118.2ms | 1.58x | <150ms ✓ |
| **Moyenne** | - | **98.8ms** | **61.4ms** | **1.63x** | **<90ms ✓** |

Les résultats démontrent un speedup consistant de 1.58-1.71x sur l'ensemble des appareils testés, respectant les objectifs de latence fixés pour chaque catégorie d'appareil. L'accélération moyenne de 1.63x valide les optimisations architecturales et confirme l'applicabilité en temps réel pour le marquage de présence.

### 2.2 Consommation mémoire et efficacité énergétique

L'analyse de l'empreinte mémoire révèle une réduction proportionnelle à la compression paramétrique, avec des bénéfices additionnels sur la consommation énergétique liée aux accès mémoire réduits.

**Tableau 5.5** : Utilisation mémoire et consommation énergétique

| Métrique | FeatherFace V1 | FeatherFace V2 | Amélioration |
|----------|----------------|----------------|--------------|
| RAM Pic (MB) | 78.4 | 43.2 | 44.9% |
| RAM Moyenne (MB) | 64.7 | 36.8 | 43.1% |
| GPU Memory (MB) | 156.3 | 89.1 | 43.0% |
| Énergie/inférence (mJ) | 142.7 | 87.3 | 38.8% |
| Temps batterie (heures)* | 3.2 | 5.1 | +59.4% |

*Usage continu détection 1 fps sur iPhone SE 2022

La réduction de consommation énergétique (38.8% par inférence) génère une augmentation significative de l'autonomie (+59.4%), facteur critique pour l'usage scolaire où la charge fréquente des appareils n'est pas toujours possible.

