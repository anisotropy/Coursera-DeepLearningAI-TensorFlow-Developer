# 1. 1-1
# xs = np.array([0., 1., 2., 3., 4., 5., 6.], dtype=float)
# ys = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5])
# y = x / 2 + 0.5
#
# 2. 1-2
# mnist = tf.keras.datasets.fashion_mnist
#
#
# 3. 2-3
# horse-or-human
# train_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
# validation_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'
#
# 4. 2-1
# cats_and_dogs_filtered
# data_url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# weights_url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
#
# 5. 2-2
# data_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
#
# 6. 2-4
# sign-mnist
# https://www.kaggle.com/datamunge/sign-language-mnist/home
#
# 7. 2-5
# rps
# train_url = https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
# valid_url = https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip
#
# 8. 3-1
# imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
#
# 9. 3-2
# sarcasm.json
# url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'
#
# 10. 3-3
# imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
#
# 11. 3-4
# bbc-text.csv
# url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv'
# stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
#
# 12. 3-5
# training_cleaned.csv
# embedding weights: glove.6B.100d.txt
#
# 13. 3-6
# data = "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."
# seed_text = "Laurence went to dublin"
#
# 14. 3-7
# irish-lyrics-eof.txt
# https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt \
# seed_text = "I've got a bad feeling about this"
#
# 15. 3-8
# sonnets.txt
# https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt
# seed_text = "Help me Obi Wan Kenobi, you're my only hope"
#
# 16. 4-1
# synthetic time series
#
# 17. 4-2
# sunspots.csv
# https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv
#
# 18. 4-3
# daily-min-temperatures.csv
# https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
