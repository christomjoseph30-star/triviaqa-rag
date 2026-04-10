# Results — TriviaQA RAG Pipeline

**Model:** jina-embeddings-v5-text-nano + gpt-4.1  |  **Recall@5:** 0.94  |  **Exact Match:** 0.86  |  **Token F1:** 0.89

| # | Question | Gold Answer | Predicted | Retrieval Hit | Exact Match | Token F1 |
|---|---|---|---|---|---|---|
| 1 | Which Lloyd Webber musical premiered in the US on 10th December 1993? | Sunset Boulevard | Sunset Boulevard | No | Yes | 1.0 |
| 2 | Who was the next British Prime Minister after Arthur Balfour? | Campbell-Bannerman | Henry Campbell-Bannerman | No | Yes | 1.0 |
| 3 | Who had a 70s No 1 hit with Kiss You All Over? | Exile | Exile | Yes | Yes | 1.0 |
| 4 | What claimed the life of singer Kathleen Ferrier? | Cancer | cancer | Yes | Yes | 1.0 |
| 5 | Which actress was voted Miss Greenwich Village in 1942? | Lauren Bacall | Lauren Bacall | Yes | Yes | 1.0 |
| 6 | What was the name of Michael Jackson's autobiography written in 1988? | Moonwalk | Moonwalk | Yes | Yes | 1.0 |
| 7 | Which volcano in Tanzania is the highest mountain in Africa? | Kilimanjaro | Kilimanjaro | Yes | Yes | 1.0 |
| 8 | The flag of Libya is a plain rectangle of which color? | Green | green | Yes | Yes | 1.0 |
| 9 | Of which African country is Niamey the capital? | Niger | Niger | Yes | Yes | 1.0 |
| 10 | Which musical featured the song The Street Where You Live? | My Fair Lady | My Fair Lady | Yes | Yes | 1.0 |
| 11 | "Who was the target of the failed ""Bomb Plot"" of 1944?" | Hitler | Adolf Hitler | Yes | Yes | 1.0 |
| 12 | Who had an 80s No 1 hit with Hold On To The Nights? | Richard Marx | Richard Marx | Yes | Yes | 1.0 |
| 13 | Who directed the classic 30s western Stagecoach? | John Ford | John Ford | Yes | Yes | 1.0 |
| 14 | Dave Gilmore and Roger Waters were in which rock group? | Pink Floyd | Pink Floyd | Yes | Yes | 1.0 |
| 15 | Which highway was Revisited in a classic 60s album by Bob Dylan? | 61 | Highway 61 | Yes | No | 0.6667 |
| 16 | Which was the only eastern bloc country to participate in the 1984 LA Olympics? | Rumania | Romania | Yes | Yes | 1.0 |
| 17 | Which 90s sci fi series with James Belushi was based on Bruce Wagner's comic strip of the same name? | Wild Palms | Wild Palms | No | Yes | 1.0 |
| 18 | If I Were A Rich Man Was a big hit from which stage show? | Fiddler on the Roof | Fiddler on the Roof | Yes | Yes | 1.0 |
| 19 | Men Against the Sea and Pitcairn's Island were two sequels to what famous novel? | Mutiny On The Bounty | Mutiny on the "Bounty" | Yes | Yes | 1.0 |
| 20 | What was Truman Capote's last name before he was adopted by his stepfather? | Persons | Persons | Yes | Yes | 1.0 |
| 21 | In Lewis Carroll's poem The Hunting of the Snark, what did the elusive, troublesome snark turn into to fool hunters? | A boojum | Boojum | Yes | Yes | 1.0 |
| 22 | What Michelle Pfeiffer movie got a boost from the Coolio song Gangsta's Paradise? | Dangerous Minds | Dangerous Minds | Yes | Yes | 1.0 |
| 23 | What was President Gerald Ford's middle name? | Rudolph | Rudolph | Yes | Yes | 1.0 |
| 24 | Art Garfunkel trained for which profession although he didn't qualify? | Architect | mathematics education | Yes | No | 0.0 |
| 25 | Who was the last inmate of Spandau jail in Berlin? | Rudolf Hess | Rudolf Hess | Yes | Yes | 1.0 |
| 26 | In the 80s who wrote the novel Empire of The Sun? | J. G. Ballard | J. G. Ballard | Yes | Yes | 1.0 |
| 27 | Kagoshima international airport is in which country? | Japan | Japan | Yes | Yes | 1.0 |
| 28 | In which sport could the Pacers take on the Pistons? | A: Basketball | Basketball | Yes | Yes | 1.0 |
| 29 | Kim Carnes' nine weeks at No 1 with Bette Davis Eyes was interrupted for one week by which song? | Stars on 45 medley | I don't know | Yes | No | 0.0909 |
| 30 | Where in Canada is the Lion's Gate bridge? | Vancouver | Vancouver, British Columbia | Yes | Yes | 1.0 |
| 31 | What was Walter Matthau's first movie? | The Kentuckian | I don't know | Yes | No | 0.0 |
| 32 | Where in England was actor Nigel Hawthorne born? | Coventry | Coventry, Warwickshire | Yes | Yes | 1.0 |
| 33 | In Greek mythology, who were Arges, Brontes and Steropes? | Cyclopes | Cyclopes | Yes | Yes | 1.0 |
| 34 | Which musician founded the Red Hot Peppers? | Jelly Roll Morton | Jelly Roll Morton | Yes | Yes | 1.0 |
| 35 | Where did the Shinning Path terrorists operate? | Peru | Peru | Yes | Yes | 1.0 |
| 36 | How old was Jimi Hendrix when he died? | 27 | 27 | Yes | Yes | 1.0 |
| 37 | Which Brit broke the land speed record in 1990 in Thrust 2? | Richard Noble | I don't know | Yes | No | 0.0 |
| 38 | In which state was Gene Vincent born? | Virginia | Virginia | Yes | Yes | 1.0 |
| 39 | How was the European Recovery Program in the 1940s more commonly known? | Marshall Plan | Marshall Plan | Yes | Yes | 1.0 |
| 40 | Brandon Lee died during the making of which movie? | The Crow | The Crow | Yes | Yes | 1.0 |
| 41 | Who had a 70s No 1 hit with Let's Do It Again? | The Staple Singers | The Staple Singers | Yes | Yes | 1.0 |
| 42 | Who had a Too Legit To Quit Tour? | MC Hammer | Hammer | Yes | No | 0.6667 |
| 43 | Which country does the airline TAAG come from? | Angola | Angola | Yes | Yes | 1.0 |
| 44 | Which US No 1 single came from Diana Ross's platinum album Diana? | Upside Down | Upside Down | Yes | Yes | 1.0 |
| 45 | River Phoenix died during the making of which movie? | Dark Blood | Dark Blood | Yes | Yes | 1.0 |
| 46 | Which artist David was born in Bradford UK? | Hockney | David Hockney | Yes | Yes | 1.0 |
| 47 | Richard Daley was mayor of which city for 21 years? | Chicago | Chicago | Yes | Yes | 1.0 |
| 48 | "In which movie did Garbo say, ""I want to be alone""." | Grand Hotel | Grand Hotel | Yes | Yes | 1.0 |
| 49 | What is Osbert Lancaster best known for producing? | Cartoons | cartoons | Yes | Yes | 1.0 |
| 50 | Who was the defending champion when Martina Navratilova first won Wimbledon singles? | Virginia Wade | Chris Evert | Yes | No | 0.0 |