# Informazioni Generali
Il progetto è stato svolto come come Attività Progettuale per il corso di Sistemi Digitali M dell'Università di Bologna.
L'obiettivo è quello di creare una applicazione per un dispositivo embedded (in questo caso uno smartphone Android) che sia in grado, attraverso l'utilizzo della fotocamera, di riconoscere una moneta, in particolare classificare le monete commemorative da 2€.  
<img src="https://github.com/gabrielecorsi97/attivita_progettuale_SistemiDigitaleM/blob/master/images/2e_nationalbank_2016.jpg" width="150" height="150">
<img src="https://github.com/gabrielecorsi97/attivita_progettuale_SistemiDigitaleM/blob/master/images/2e_italy_2012.jpg" width="150" height="150">
<img src="https://github.com/gabrielecorsi97/attivita_progettuale_SistemiDigitaleM/blob/master/images/2e_torino_2006.jpg" width="150" height="150">
<img src="https://github.com/gabrielecorsi97/attivita_progettuale_SistemiDigitaleM/blob/master/images/2e_donquixote_2005.jpg" width="150" height="150">  
Questa funzionalità è stata implementata utilizzando una rete neurale convoluzionale (o CNN), in particolare una rete di similarità ([similarity learning](https://en.wikipedia.org/wiki/Similarity_learning)). 
Durante l'addestramento la rete impara a creare un vettore (chiamato embedding) per ogni immagine in input tale per cui immagini della stessa classe abbiano embedding spazialmente vicini, mentre embedding appartenenti a classe diverse siano spazialmente lontani.
La dimensionalità degli embedding scelta per il progetto è 128.
L'immagine seguente mostra l'embedding space "compresso" in 3 dimensioni. Ogni colore indica una classe del dataset diverso.  

<img src="https://github.com/gabrielecorsi97/attivita_progettuale_SistemiDigitaleM/blob/master/images/embedding_space.gif" width="600" height="600">  

## Demo  
<img src="https://github.com/gabrielecorsi97/attivita_progettuale_SistemiDigitaleM/blob/master/images/demo_app.gif"  height="400">  

## Report
Per ulteriori dettagli è possibile visionare [il report del progetto](https://github.com/gabrielecorsi97/attivita_progettuale_SistemiDigitaleM/blob/master/RelazioneAttivit%C3%A0ProgettualeSistemiDigitaliM.pdf).  
## Librerie Utilizzate
Le librerie utilizzate sono state: 
  - Tensorflow 2.13.0
  - Tensorflow Similarity 0.17.1
  - Tensorflow Model Maker 0.4.2


