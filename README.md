**Overview**: 

This system generates comedic stand-up routines. It is called the "Stand Up Stooge." It uses a neural network that implements long short-term memory (LSTM) units to learn from Seinfeld's previous routines, and then generates new routines. Learning implements early stopping (based on loss values ceasing to decrease) and check-pointing. Ten example outputs can be seen in the 'examples' folder. The '|' symbol indicates the change from seed text to generated text. 
`all_routines.txt` contains multiple generations from training on routines on four different comedians (including Seinfeld). As there was significantly more text to train on, it seems that the model did not have enough predictive power, as the output is highly repetitive (discussed further below). `_learnMultiple.py` was used to train this model (not included here).

Much of the learning and generation code was borrowed from these two Keras tutorials, although was modified to suit my needs: [(analyticsvidhya.com)](https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/) [(machinelearningmastery.com)](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)

---

**Setup**:

Running `generate.py` will use the pre-trained model to generate 1000 characters of new material, using a random 250-character segment of the input (pre-existing) text as a seed input.  
Major dependencies are:  
– Keras  
– numpy

---

**System Architecture**:

This system focuses on comedy generation. Previous works in this realm have focused on joke formation, particularly using strict joke 'formulas' (e.g. [Petrovic and Matthews](http://www.aclweb.org/anthology/P13-2041)). They have also commented, however, that it is difficult to describe why some jokes "work" and others do not. This system instead attempts to model humor by creating short stand-up skits with comedic value. The hope is that, by studying enough comedic content, the system will be able to produce unique content that is funny, without a structured understanding of why the product is funny.

Collection of previous stand-up material was done largely by hand. Stand up routines from the show *Seinfeld* were taken from scripts found here: [IMSDB – Seinfeld](https://www.imsdb.com/TV/Seinfeld.html). The remaining stand-up routines were pulled from [Scraps from the Loft](https://scrapsfromtheloft.com/tag/stand-up-transcripts/). This latter site seems more capable of being scraped using an automated method, if collecting much more inspiring routines was desired in the future.
All content came from Jerry Seinfeld, John Mulaney, Sarah Silverman, or Trevor Noah.

Training was done using a Keras neural network in `learn.py`. Characters (a-z, ' ', '\n') were tracked using one-hot encoding. The model uses an initial embedding layer, two LSTM layers with dropout after each, and a dense layer before the SoftMax output layer (See summary below). This structure was inspired by online resources and finalized after iterative testing of designs. The model learned on 250 characters of the input set, with the expected output being the next character in the text. It ran for 50 epochs, or until loss stopped decreasing.

| Layer (type)            | Output Shape     | Param # |
|:-----------------------:|:----------------:|:-------:|
| embedding_1 (Embedding) | (None, 250, 16)  | 464     |
| lstm_1 (LSTM)           | (None, 250, 500) | 1034000 |
| dropout_1 (Dropout)     | (None, 250, 500) | 0       |
| lstm_2 (LSTM)           | (None, 500)      | 2002000 |
| dropout_2 (Dropout)     | (None, 500)      | 0       |
| dense_1 (Dense)         | (None, 200)      | 100200  |
| dense_2 (Dense)         | (None, 29)       | 5829    |

Once the model was trained, it is used to generate new stand-up routines (`create.py`). By default, the generator picks a random sequence from the inspiring set and uses it to generate the next character of the output text. The generator repeats this process, using the last segment of the growing output text as the input of the model. The generator can also use user input as the initial input seed.

Valid calls to the generator:  
`python3 create.py <'seedText'> <outputLength>`  
`python3 create.py <'seedText'>`  
`python3 create.py`

If seedText is `'random'`, the seed will be set to a random combination of (a-z, ' ', '\n') characters.

*Fig. 1* – Diagram of system
![Figure 1, diagram of system](/structure.png "System Diagram")
The source text is fed into the learning algorithm, which models the probability of a character following a given series of previous characters. The trained model is then used, along with seed text, in the generator to create novel comedic stand-up routines as output. 

---

**Computational Creativity**:

The system strives for both novelty and value, which are common goals in computational creativity study ([Jordanous](https://link.springer.com/article/10.1007/s12559-012-9156-1)). As stated previously, however, it is not easy to describe how or why something is funny. This complicates evaluating value. I decided to simply use my own personal judgement in deciding how funny the output was, especially with relation to how funny the input was. I chose to evaluate the system primarily on the *Generation of Results* and *Independence and Freedom* SPECS criteria.

In terms of *Generation of Results*, the system is only moderately successful. I would consider a complete success as output that is logical (in terms of grammar, syntax, etc.) without simply reproducing content from the original routines used as the inspiring set. Unfortunately, I found it difficult to balance these two factors. The output obtained through various trials of 'window' size (the length of the input string), model specification, and input changes (including/excluding punctuation) resulted in one of either of these faults. Very often (especially with under-trained/weak models or very (50 characters) short input windows) the output would become a repeating series of some number of words. Otherwise, the output would replicate the input set to some degree (often this would entail the output being composed of chunks of pre-existing routines, with sudden jumps between them). Both of these disappointing results can be seen in the generated output in the 'examples' folder.

The other criteria I set out to achieve is Independence and Freedom. This focuses more particularly on the novelty of the system output. I hoped that having a character-level model (as opposed to word-level) would encourage novelty. In theory, any string of the allowed characters is feasible. However, this restriction of limiting the output to what is the input is made of is a limitation I did not fully understand when planning. I thought that removing punctuation and other marks (as said before, only allowing for a-z, ' ', and '\n') would have multiple benefits.  
Most simply, limiting the options decreases the size of the problem, as there are fewer input and output possibilities. On evaluation, I maintain that this is true, as removing other symbols reduced the model size and increased training speed.  
Additionally, I thought that making the inspiring text more similar by removing other 'rarer' characters would enable more holistic learning of general sentence and joke structure, rather than learning the exact construction of individual sentences in the inspiring set. It now seems like this is not true. Perhaps having more input variation would have led to a similarly varied output.  
Finally, removing all punctuation also negatively impacts the understandability of the output. Much of stand-up is situational comedy. This often requires quoting other people. By removing quotation marks from the system's vocabulary, it loses the concept of fluidly incorporating quotations in stand-up routines, and the resulting text output is much less clear (as it is difficult to tell if the text is the system 'speaking' or 'quoting' something/one else).

---

**Personal Challenges**:

Learning how to use Keras was challenging but really rewarding as I am exciting to use it again in the future. I really enjoyed exploring a different learning algorithm than I had used previously (like Markov chains and genetic algorithms). It was also really cool getting to use the HPC grid to train the models. I was disappointed that the results were not better than they were, especially when using stand-up routines from multiple comedians. This was something I thought of towards the end of the project, and if I had more time I would train a larger model on this data to hopefully produce better output, as I think it would be interesting to see if the output merged the styles of the various comedians. Despite the shortcomings in the generated text, it was a rewarding process to analyze the output on computational creativity terms.
