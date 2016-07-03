TEST datasets for NLPCC2016 Task 4 Stance Detection in Chinese Microblogs

INPUT DATA FORMAT

The test datasets have the following format (the same format that was used in the training data):
<ID><tab><Target><tab><Tweet><tab>UNKNOWN

where
<ID> is an internal identification number;
<Target> is the target entity of interest (e.g., "Hillary Clinton"; there are five different targets in task A and one target in Task B);
<Tweet> is the text of a tweet.


SUBMISSION FORMAT

Your submission for each task should include two files:
1. prediction file
2. system description file

The prediction file should have the same format as the test file; just replace the word UNKNOWN with a predicted stance label. Please keep using a TAB (NOT a SPACE) as the delimiter between different columns, as in the original test file.

The possible stance labels are:
1. FAVOR: We can infer from the text that the user supports the target (e.g., directly or indirectly by supporting someone/something, by opposing or criticizing someone/something opposed to the target, or by echoing the stance of somebody else).
2. AGAINST: We can infer from the text that the user is against the target (e.g., directly or indirectly by opposing or criticizing someone/something, by supporting someone/something opposed to the target, or by echoing the stance of somebody else).
3. NONE: none of the above.

The system description file should provide a short description of the methods and resources used in the following format:

1. Team ID
2. Team affiliation
3. Contact information
4. System specifications:
- 4.1 Supervised or unsupervised
- 4.2 A description of the core approach (a few sentences is sufficient)
- 4.3 Features used (e.g., n-grams, sentiment features, any kind of tweet meta-information, etc.). Please be specific, for example, the exact meta-information used.  
- 4.4 Resources used (e.g., manually or automatically created lexicons, labeled or unlabeled data, any additional set of tweets used (even if it is unlabeled), etc.). Please be specific, for example, if you used an additional set of tweets, you can specify the date range of the tweets, whether you used a resource publicly available or a resource that you created, and what search criteria were used to collect the tweets. 
- 4.5 Tools used
- 4.6 Significant data pre/post-processing
5. References (if applicable)

These descriptions will help us to summarize the used approaches in the final task description paper.

You can provide submissions for either one of the tasks, or both tasks.


EVALUATION

System predictions will be matched against manually obtained gold labels for all instances in the test sets. We will use the macro-average of F-score(FAVOR) and F-score(AGAINST) as the bottom-line evaluation metric.