#Lernplan NN

## Themenschwerpunkte nach Protokollen

+ Was ist ein NN? [Q036640]
+ Für was kann man NNs einsetzen? [Q036640]

+ Klassifikation
	+ parametrisch / nicht-parametrisch
		+ KNN, parzen window
	+ überwacht / unüberwacht
		+ Überwachtes / unüberwachtes Lernen
			+ Reinforcement Learning
		+ LVQ1, LVQ2, LVQ3
		+ Kohonens SOM
	+ linear / nicht-linear

+ Perzeptron
	+ Aufbau
	+ Fehlerfunktionen
		+ classification figure of merit
	+ Aktivierungsfunktionen
		+ radial basis function kernel ??
	+ Gewichte einstellen
		+ Backpropagation
		+ (L-)BFGS
		+ Conjugate Gradient
		+ Quasinewton
	+ Lernen / Training
		+ Wie kann ich das Training beschleunigen? [Q010701]
			+ Momentum
			+ Quickpop
			+ Lernrate anpassen
		+ Lernrate einstellen
			+ Momentum Methode
			+ Nesterov Momentum
			+ Scheduling Methode
			+ NewBob Methode
			+ Exponentially Decying
			+ gewichtsabhängige Methodne
				+ AdaGrad
				+ Resilient Propagation
				+ Weiterentwicklung Mean Square Resilient Propagation
		+ Hebbian Lerning
	+ "Entscheidunsoberfläche"?? [Q009637]
	+ XOR-Problem
	+ MLP
		+ Aufbau
		+ Lernen
			+ Kettenregel
		+ Was ist entscheidend für die Nichtliniarität von MLPs? [Q009620]
		+ Autoencoder

+ Was sind Elman und Jordan Netze? [Q002300]
	+ Wie trainiert man sie?

+ Spracherkennung
	+ Einsatz von NNs im Spracherkenner? [Q008708]
		+ Zum Schätzen der Emissionswahrscheinlichkeit in HMM
		+ Was gibt das NN für Wahrscheinlichkeiten aus? [Q008708]
			+ A-posteriori [Q011708]
		+ Vorverarbeitung dieser Wahrscheinlichkeiten vor Einsatz in HMM nötig?
		[Q008708]
			+ müssen für die Emissionswahrscheinlichkeit in klassenbedingte
			 Wahrscheinlichkeit umgerechnet werden (Bayes) [Q011708]
			+ "Fundamentalformel"?? [Q008708]

+ Generalisierung
	+ Was führt zu besserer Generalisierung? [Q008708]
	+ Overfitting
	+ Optimal Brain Surgeon ?? [Q008708]
	+ Meiosis Netze ?? [Q008708]
	+ Regulierung von Gewichten
		_weight elimination_, _weight limitation_, _weight decay_, _dropout_
	+ _OBD_, _OBS_
	+ cascade correlation, weight elimination, weight decay
	+ Wofür braucht man Crossvalidation-Daten? [Q011708]x
	+ Fehlerkurven
		+ Was würden Sie hier raten? [Q002300]
			+ Komplexität des Netzes anpassen (Anzahl der Hidden Units).
			+ Netzstruktur anpassen.
			z.B. _cascade correlation_, _mesios_
		+ Wann hört man auf mit Trainieren? [Q002300]
		+ Lernrate zu groß / zu klein [Q009620], [Q011708]
	+ `E_test = E_train + 2 o * p / n` [Q025194]
		+ Wie findet man ein geeignets `p`? [Q025194]

+ Backpropagation
	+ Warum kein Simulated Anneahling notwendig? [Q011708]
	+ Wie kann man Backpropagation verbessern? [Q026085]
	+ versch. Arten des Trainings mit BP [Q025194]
		+ BP nach jedem Lernbeispiel
		+ BP nach einem Block von Lernbeispielen, der von jeder Klasse 
		ein Beispiel enthält
		+ BP nach allen Lernbeispielen
	+ Beschleunigungsverfahren von BP [Q025194]
		+ Momentum, Quickprop, Resilent Propagation
	
+ Bolzmann-Machine
	+ Simulated Anealing [Q011708]
	+ Was unterscheidet Bolzmann-Machines von Backpropagation Netzen? [Q026085]
	+ Wann sind sie stochastisch? [Q026085]

+ Was ist das Problem bei vielen Klassen? [Q026085]
	+ classification figure of merit

+ Wie integriert man verschiedene Netze? [Q026085]
	(z.B. TDNNs bei Spracherkennung)
	+ _connectionist glue_
	+ Dynamic-Time-Warping-Layers

+ Wie funktionieren hybride Verfahren? [Q026085]
	+ Kombination von NN mit andern ML-Verfahren. Z.B. HMMs

+ Was sind modale Netze? [Q026085]

+ Wenn ich vertauschte Daten habe, was mache ich da? [Q035940]
	+ Denoising Autoencoder

+ Was ist der Vorteil der Momentum Methode gegenüber der Scheduling Methode?
 [Q036640]

+ LVQ

+ Hopfield Nets

+ TDDN
	+ weight sharing
