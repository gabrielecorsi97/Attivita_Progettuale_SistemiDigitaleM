# Informazioni Generali
Il progetto è stato svolto come come Attività Progettuale per il corso di Sistemi Digitali M dell'università di Bologna.
Il progetto ha come obiettivo quello di creare una applicazione per un dispositivo embedded (in questo caso uno smartphone Android) che sia in grado, attraverso l'utilizzo della fotocamera, di riconoscere una moneta, in particolare classificare le monete commemorative da 2€.
Questa funzionalità è stata implementata utilizzando una rete neurale convoluzionale (o CNN), in particolare una rete di similarità. 
Durante l'addestramento la rete impara ad creare un embedding per ogni immagine in input tale per cui immagini della stessa classe abbiano embedding spazialmente vicini, mentre embedding appartenenti a classe diverse siano spazialmente lontani.
La dimensionalità degli embedding scelta per il progetto è 128.
## Librerie Utilizzate
Le librerie utilizzate sono state: 
  - Tensorflow 2.13.0
  - Tensorflow Similarity 0.17.1
  - Tensorflow Model Maker


