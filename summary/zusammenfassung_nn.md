#Zusammenfassung NN

### Warum NN?
Was kann unser Gehirn besonders gut?
**Abstraktion** (Beispiel Buchstabe "A" [S.2] => Klassifikationsproblem)

### Klassischer Ansatz vor NN:
+ Regeln definieren (Maschine soll nicht selber lernen)
+ Schablonen mit Distanzfunktion

Von Neumann Ansatz: Eine zentrale Recheneinheit. Anweisungen und Daten liegen im gleichen Speicher (Gegensatz zu Harward Architektur). Anweisungen werden sequentiell abgearbeitet. Kommunikation schlecht, weil keine/wenig Kommunikationspartner (Komm. nur über Speicher möglich). Bei NN sind Neuronen vernetzt. NNs leben von Kommunikation. NN kann nur bestimmte Aufgaben lösen, von Neumann ist dagegen universell einsetzbar. (Vgl. Tabelle S.4)

Künstliche NN sind von der Biologie inspiriert, sind aber nicht das Selbe. NN rechnen bspw. _scharf_ (richtig) im Gegensatz zu Neuronen im menschlichen Gehirn. Hardware auf der NNs laufen ist nicht ideal für NN. Parallelisierung erst auf z.b. GPUs.

*Anwendungen* [S.5]: z.B. Buchstabenerkennung.
Dem gegenübergestellt, das *Problem*: Klassifikationsproblem.

Roboter hat Perzeptronen um Umwelt wahrzunehmen und Aktoren um Einfluss auf sie nehmen zu können. NN sind bspw. geeignet um Roboter laufen zu lassen.

NN für Genre-Erkennung bei Musik, oder zum Komponieren.

### Low level problems
1. Klassifikations
2. Regression / Funktionsapproximation
3. Sequenzgenerierung
4. Clustering
5. Merkmalsextraktion (Vorverarbeitung)
6. Assoziative Speicher (z.B. mit SOMs)
7. Kontrolle von Aktuatoren
8. Vorhersagen

### NN Models
siehe Folie 9

### Bolzmann Machine vs. Hopfield Net
Beides sind rekurrente Netze (Verbindungen sind immer bidirektional verknüpft.). Beide haben zu einem Zeitpunkt eine bestimmte Belegung. Werte im Hopfield Netz sind binär.

Wie berechnet sich der Wert eines Neurons?

+ Hopfield: deterministisch: Ist (Eingänge + Gewichte) > oder < Threshold? => Ergebnis: -1 oder 1
+ Bolzmann machine: Ob ein Wert auf +1 oder -1 gesetzt wird ist nicht mehr deterministisch. Stattdessen wird eine Wahrsch. ausgerechnet, mit der ein Neuron auf 1 oder -1 gesetzt wird. Wert einer Einheit ist damit abhängig von Zufall. Es kann nur mit einer Wahrscheinlichkeit der Wert vorrausgesagt werden.

Update:

1. sequentiell: zufällige Einheit berechnen
2. gleichzeitig: Alle Elemente werden auf einmal geupdated


Hopfield: Je mehr Iterationen, desto kleiner ... => Konvergiert in ein lokales Minimum in der Energie.

Was nützen diese Netze?

+ Hopfield Netz hat eine bestimmte Energie: Nützlich für assoziative Speicher. Bei neuem Muster solange iterieren, bis in lokalem Energieminimum. Wenn das Minimum gleich dem Minimum ist, welches bspw. mit Katzenbildern trainiert wurde, handelt es sich um ein Katzenbild.
+ Bolzmann: B. Netz hat versteckte und sichtbare Einheiten (Bei H. gleichverteilt). Unsichtbare Neuronen sind etwas wie _versteckte Wissensrepräsentanten_. Rekonstruktion der Werte der Hidden Units bei lokalem Minimum möglich.

Nachteil bei Bolzmann machines: Bei Trainieren ist der Trainingsalgorithmus extrem langsam. Aufwand wächst exponentiell mit der Anzahl der Units. Für einfache (wenige Units) Probleme gut geeignet, sonst viel zu langsam und wenige Trainingsdaten. (kein Mensch verwendet sie, trotz gutem Konzept.)

Lösung: Abwandlung (statt Bolzmann machine): restricted bolzmann machine. Innerhalb der visible Units sind die Units nicht miteinander verbunden. Ebenso bei hidden Units.

### Autoencoder
Ausgabegröße = Eingabegröße (vollverbunden)

**Ziel**: Muster anlegen, Netz soll lernen, dass das selbe am Ausgang herauskommt.
Eingabe wurde mit Rauschen korrumpiert. Netz soll lernen das Rauschen zu entfernen. Deshalb wird die Eingabe mit Rauschen vermischt.

Welche Arten von Noise gibt es?

+ Rauschen
+ Neuronen auf 0 setzen (simuliert verlorengegangene Neuronen -> SPARSE Autoeencoder)

Welches Problem kann damit noch (ein bisschen) gelöst werden? Was denoising autoencoder haben mit deeplearning zu tun?

Der Gradient bei Backpropagation wird flach. => Viele Iterationen notwendig, um zum Minimum zu kommen.

Wie kann dieses Problem abgemildert werden?

+ Gewichte vortrainieren (Initalisierung in sinnvoller Region)
	+ Backpropagation verbessert Initalisierung (die sonst zufällig gewählt wurde)

Wie geht Vortraining mit denoising autoencoder?

+ Trainingsdaten werden angelegt und schichtweise trainiert (beginnend bei erster Schicht)

	+ 1. Schicht lernt
	+ 2. Schicht wird drauf gesetzt und wieder trainiert
	+ usw.

Weitere Möglichkeit des Vortrainierens:

+ Restricted Bolzmann Machine (2 Layer) => liefert Gewichte

### Bottleneck Features
Dienen der Vorverarbeitung (Dimensionsreduktion). Z.B. notwendig bei Vorverarbeitung von Klassifikationsprobleme.

Hat erst mal nichts mit Klassifikation zu tun. Gesucht sind Merkmale! Klassifikationsproblem des Bottleneck soll den Merkmalen des späteren Netzes entsprechen. Netz trainieren welches den Zustand HMM klassifiziert. => Merkmale so transformieren, dass sie zu den Klassen des HMM passen.

### Aktivierungsfunktion
Bei Perzeptronen! Rosenblatt-Perzeptron hat {0,1} als Ausgabe (je nach Schwellwert).

#### Softmax
Warum macht die softmax Funktion bei Perzeptronen keinen Sinn?

+ Bsp: 10 Neuronen mit Werten liefert eine Wahrscheinlichkeitsverteilung. Macht nur Sinn, wenn mehrere Ausgabeneuronen, sonst Ausgabe immer gleich 1.

Ist geeignet für Klassifikation, weil a posteriori Wahrscheinlichkeiten bei geeigneter Fehlerfunktion (Mean-squared-Error?)

#### Linear Function
MLP mit linearer Aktivierungsfunktion macht keinen Sinn, weil wieder auf ein Perzeptron reduzierbar. MLPs sollen ja genutzt werden, um nicht linear separierbare Probleme zu lösen.

#### Maxout
Erhöht die Generalisierungsfähigkeit des Netzes.



# Übersicht A. Waibel
Ziel der VL: Grundlegende Modelle kennenlernen, Applikationen welche NN zum Einsatz bringen. Verstehen, das mit den Netzen versch. Dinge gemacht werden können: Klassifikation, Vorhersage, Encoder

+ Backpropagation herleiten u. programmieren können
+ Simulatoren mit Grafiken, etc.
+ Bolzmann machines, LVQ, Hopfield Netze (vorfahren von Bolzmann machines)

+ Wie funktionieren die Algorithmen?
	+ Backpropagation __muss__ gekonnt werden!

+ Grundlagen der Mustererkennung
+ Backpropagation im Einsatz
	+ Was ist Generalisierung
	+ Hoch komplexer Klassifikator, der vieles kann. Denn in den Begriff zu bekommen, ist eine Kunst.
	+ Overfitting / Generalisierung
	+ Gegenteil: Zu kleines Netz: Das Netz wird nichts lernen. (Generalisierung ist gut, weil Training und Test sehr schlecht.)

+ Lernkurven sind wichtig in der Praxis (sollten während des Trainings beobachtet werden.)
	+ Fehlerfunktionen messen
	+ Fehler sollten über die Trainingszeit weniger werden.
	+ Ist das nicht der Fall: Netz größer und weniger Daten.
	+ Denn zwei Daten und großes Netz: Dann muss der Fehler irgendwann runter gehen. Das Netz wird eine Trennlinie finden. Sonst ist etwas im Training falsch. Z.B. Schrittgröße schlecht gewählt. -> Kontrolle der Schrittgröße wichtig
	+ Anfang: Schrittgröße groß
	+ Schrittgröße (Epsilon) vs. Momentum turn (alpha)
	+ Gradientdecent-Verfahren
	+ Anfangs: wenige Daten, dann nach und nach Datenmenge größer
	+ Trainings- und Testkurve sollen möglichst nah beieinander sein: Generalisierung
	+ __Was ist Generalisierungsfähigkeit und wie funktioniert das Training.__ (sehr wichtig!)
		+ Wie ist die richtige Schrittgröße, um das Minimum zu finden.
		+ Welche Werte für die Schrittgröße.

+ versch. Fehlerfunktionen, welche gibt es? Softmax, Cross-entropie, MSE, ...

+ Automatische Methoden um Minima zu finden (Training).
	+ Beschleunigung durch automatisches Anpassen der Schrittgröße.
	+ konstrukt. und destruktive Lernalgorithmen		+ cascading ...

+ RNN
	+ sequenzen von Informationen und Zeitverläufe
	+ Lippenlesen, Spracherkennung, Handschrifterkennung
		+ TDNN (eindimensional über die Zeit, Sprecherunabhängikeit: zweidimensional)
		+ shift invarianz in unterschiedlicher Dimension (Zeit, Frequenz)
			+ Irrelevante _Daten_ wegbekommen
+ Sequenzen
	+ Rekurenzen (Neuron lernt eine gewisse Historie mit)
		+ rückwärts aufrollen ist kompliziert und rechenaufwändig
	+ Streit: statt Rekurenz auch n-Tupel verwenden

+ Zustände
	+ hybride Ansätze
		+ Kombination mit Markov Kette oder HMM
		+ HMM mit NN als Klassifikatoren

		+ Was ist bei einem hybriden Ansatz zu beachten: __Normierungen__!
			+ NN: a posteriori Wahrscheinlichkeit
			+ HMM: klassenbedingte Wahrscheinlichkeiten

+ zwei, drei Modelle erklären können, wie in der Spracherkennung NN eingesetzt werden
	+ Fourrierkoeffizienten, o.ä. müssen in NN-VL nicht beherrscht werden.
	+ Es reicht zu wissen, dass es Merkmale gibt, die zeitabhängig sind.
	+ Es geht darum Sequenzen zu erkennen.
		
# Übersicht Thanh Le-Ha

+ Error Functions
+ Training tricks
	+ batch: stabiler
	+ online: schneller


