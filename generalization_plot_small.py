import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys


# Check whether the name of the pickle file  
# was provided as an argument
if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file containing data to generate the plot"
    exit(0)



def pg_barplot(pg_list, title):
    """
    Creates a histogram to visualize the probability
    of generalization and saves it in a directory
    
    Parameters
    ----------
    pg_list: list
        A list containing the probabilities of generalization for an 
        example 'y' from the subordinate, basic and superordiante categories
    title: string
        The title fo the plot (also used as filename to save the plot)
    """
    
    ind = np.arange(len(pg_list))
    width = 1                       
    plt.figure(figsize=(5, 6), facecolor='white')
    plt.bar(ind, pg_list, width, color='grey') #, yerr=menStd)
    plt.xlabel('Categories', fontsize=11)
    plt.ylabel('Probability of generalization', fontsize=12)
    plt.title(title)
    plt.xticks(ind + width/2., ['sub', 'basic', 'super'])
    #plt.show()
    plt.savefig('./generalization_plots_small/' + title, fontsize=12)
    plt.close()
    return plt
    
    
# Read the data from the pickle file 
data = pickle.load(open(fname, 'rb'))

left_branch_1sub = data['left_branch_1sub']
left_branch_3sub = data['left_branch_3sub']
left_branch_3basic = data['left_branch_3basic']
left_branch_3sup = data['left_branch_3sup']

right_branch_1sub = data['right_branch_1sub']
right_branch_3sub = data['right_branch_3sub']
right_branch_3basic = data['right_branch_3basic']
right_branch_3sup = data['right_branch_3sup']


# Average the data from the two clusters
sub1 = [sum(x)/2.0 for x in zip(left_branch_1sub, right_branch_1sub)]
sub3 = [sum(x)/2.0 for x in zip(left_branch_3sub, right_branch_3sub)]
basic3 = [sum(x)/2.0 for x in zip(left_branch_3basic, right_branch_3basic)]
sup3 = [sum(x)/2.0 for x in zip(left_branch_3sup, right_branch_3sup)]


# plot the averaged generalization results
pg_barplot(sub1, '1 sub')
pg_barplot(sub3, '3 sub')
pg_barplot(basic3, '3 basic')
pg_barplot(sup3, '3 sup')















