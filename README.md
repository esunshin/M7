**Overview**: 1 paragraph that describes your system. State the system's name and why it's called so, as well as the main components of the system and the algorithms behind them.

This system generates comedic stand up routines. It is called comedi

It uses a neural network that implements long short-term memory (LSTM) units to learn from Seinfeld's previous routines, and then generates new routines.

===

**Setup**: Step-by-step instructions for how to run your code, including any dependencies, versions, and so on.

Running `generate.py` will use the pre-trained model to generate 1000 characters of new material, using a random (HOW LONG?) segment of the input (pre-existing) text as a seed input. 

Major dependencies are:
    Keras
    numpy

===

**System Architecture**: A more detailed account of your system (at least 4 paragraphs) and its components. You should clearly describe what components of script generation that you chose to focus on (e.g., agent personality modeling, dialogue generation, narrative prose generation, suspense modeling, conflict modeling, musical lyric generation, humor and sarcasm modeling, visual animation, computational cinematography…), citing scholarly work as appropriate. Include a block diagram of your system architecture.

This system focuses on comedy generation. Previous works in this realm have focused on joke formation, particularly using strict joke 'formulas' (e.g. [http://www.aclweb.org/anthology/P13-2041]). This system instead attempts to create short stand-up skits with comedic value. The hope is that, by studying enough comedic content, the system will be able to produce unique content that is funny, without a structured understanding of why the product is funny.

Collection of previous stand-up material was done largely by hand. Stand up routines from the show *Seinfeld* were taken from scripts found here: [https://www.imsdb.com/TV/Seinfeld.html]. The remaining stand up routines were pulled from [https://scrapsfromtheloft.com/tag/stand-up-transcripts/]. This latter site seems more capable of being scraped using an automated method, if collecting much more inspiring routines was desired.
All content came from Jerry Seinfeld, John Mulaney, Sarah Silverman, or Trevor Noah.

Training was done using a Keras neural network. Characters (a-z, ' ', '\n') were coded using one-hot encoding. The model uses an initial embedding layer, two LSTM layers with dropout after each, and a dense layer before the output layer (See summary below). This structure was inspired by online resources and finalized after repeated testing. The model learned on 250 characters of the input set, with the expected output of the next character in the text. It ran for 50 epochs, or until loss stopped decreasing.

| Layer (type)            | Output Shape     | Param # |
|-------------------------|------------------|---------|
| embedding_1 (Embedding) | (None, 250, 16)  | 464     |
| lstm_1 (LSTM)           | (None, 250, 500) | 1034000 |
| dropout_1 (Dropout)     | (None, 250, 500) | 0       |
| lstm_2 (LSTM)           | (None, 500)      | 2002000 |
| dropout_2 (Dropout)     | (None, 500)      | 0       |
| dense_1 (Dense)         | (None, 200)      | 100200  |
| dense_2 (Dense)         | (None, 29)       | 5829    |

Once the model was trained, it is used to generate new stand up routines.


===

**Computational Creativity**: You should follow the general SPECS procedure to evaluate your system. Start by stating your assumptions and definitions for what it means for a system to be creative here. These statements should be founded on prior work (i.e., you should be citing respected scholars in the field). Next, clearly state at least one creativity metric for your system and evaluate it based on that metric. This metric can be derived from the SPECS themes, Ritchie's criteria, the Four PPPPerspectives, or another formalized evaluation procedure from scholarly work (e.g. Colton's Creative Tripod). Regardless of the metric and definitions you specify, you must acknowledge any limitations, biases, or potential issues with your evaluation. Your grade will not be affected if your data is biased or limited unless you leave out this information.



===

**Personal Challenges**: Describe how you personally challenged yourself on this assignment as a computer scientist. How did you strive to make your system unique, meaningful, and use sophisticated techniques? How did you push yourself as a scholar and a programmer? What new techniques did you try? What discoveries and connections did you make?






thoughts to include:
    stripped out quotation marks --> lose the comedian using quotes in a situation
        --> can be inferred, but not super easy