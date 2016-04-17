#access a method from another file
#from wl_helper import *
import numpy as np
#import pymc3 as pm
import matplotlib.pyplot as plt
import collections
import operator    # used to get the key having maximum value in a dictionary
import csv

"""
        [o]          (height = 1)
         |
        [a]          (height = 0.8)
      /    \
    /        \
  [b]       [e]        (height = 0.3)
  / \       / \
[c] [d]   [f] [g]        (height = 0)
 1  4      5   7
 2         6   8
 3             9
 
"""
################################################# Global Variables ###########################################

#constant parameter epsilon  - increasing its value give more and more basic level bias
epsilon = 0.0 #0.10 - gives around 9% more basic level bias    

#nodes = ['a','b','c','d','e','f','g']
#heights = {'o':1, 'a':0.8, 'b':0.3, 'e':0.3, 'c':0, 'd':0, 'f':0, 'g':0}
#parents = {'o':False, 'a':'o', 'b':'a', 'e':'a', 'c':'b', 'd':'b', 'f':'e', 'g':'e',
#              1:'c', 2:'c', 3:'c', 4:'d', 5:'f', 6:'f', 7:'g', 8:'g', 9:'g'}
# Global mapping from numbers to letters (node representations)
#node_map = {0: 'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g'}


def read_csv(filename):
    global nodes
    global heights
    global parents
    global node_map
    nodes = []
    heights = {}
    parents = {}
    node_map = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if i==0:
                i = i+1
                continue
            # if the node name is an integer, then it is an example/data
            elif row[0].isdigit():
                parents[int(row[0])] = row[2]
            else:
                heights[row[0]] = float(row[1])
                parents[row[0]] = row[2]
                if i!=1:
                    node_map[int(row[3])] = row[0]
                    nodes.append(row[0])
                
                i = i+1

##############################################################################################################


#check whether node1 is a descendent of node2    
def is_child(node1, node2):
    if (node1 == 'FALSE'):
        return False
    elif (node1 == node2):
        return True
    else:
        return is_child(node_parent(node1), node2)
        
def node_parent(node):
    return parents[node]

def node_height(node):
    return heights[node]
    

# function to calculate the prior of all the nodes   
# mainly need a list for multinomial distribution
def cal_prior(nodes):
    # define prior probabilities
    raw_weights = [ (node_height(node_parent(x)) - node_height(x)) for x in nodes]
    #raw_weights = [1 for x in nodes]

    # normalize prior probabilities
    weights = [float(x) / sum(raw_weights) for x in raw_weights ]
    return weights


def raw_prior(node):
    return node_height(node_parent(node)) - node_height(node)
    

def prior(node):
    # sum of priors of all nodes
    prior_denom = sum([raw_prior(node_x) for node_x in nodes])
    # return the prior of the given node
    return raw_prior(node)/prior_denom


def raw_likelihood(node, data):
    n = len(data)
    return ((node_height(node) + epsilon)**-n)


# returns normalized likelihood
def likelihood(node, data):
    # sum of likelihoods of all nodes
    likelihood_denom = sum([raw_likelihood(node_x, data) for node_x in nodes])
    # return the likelihood of the given node
    return raw_likelihood(node, data)/likelihood_denom
     

def test():
    sample_size = 1
    n = 1
    y = np.where(np.random.multinomial(n, weights))[0][0]
    print y
    print nodes[y]
    
    model = pm.Model()
    with model:
        #p = weights
        #y = [likelihood(node) for node in nodes]
        hypothesis = pm.Multinomial(name="hypothesis", n=sample_size, p=weights)
        
        start = pm.MCMC([weights, draws])
        step = pm.Metropolis()
        trace = pm.sample(1000, step, start)
    
    plt.hist(trace['p'], 15, histtype='step', normed=True, label='post');
    x = np.linspace(0, 1, 100)
    plt.plot(x, stats.beta.pdf(x, alpha, beta), label='prior');
    plt.legend(loc='best');

 
def barchart(samples):
    if sum(samples) == 0:
        print "No samples collected"
        return plt
    norm = [float(i)/sum(samples)*100 for i in samples]
    #norm2 = [norm[i] for i in np.nonzero(norm)[0]]
    #print "Norm2: ", norm2
    ind = np.arange(len(samples))
    width = 1
    plt.figure(figsize=(len(nodes), 6), facecolor='white')
    plt.bar(ind, norm, width, color='r') #, yerr=menStd)
    plt.ylabel('Probabilities')
    plt.title('Nodes')
    #plt.xticks(ind + width/2., ('a', 'b', 'c', 'd', 'e', 'f', 'g'))
    plt.xticks(ind + width/2., nodes)
    return plt


############################################## Coin toss ########################################################

#bias is the prob of 1
# like flipping a coin with prob 0.5 of heads
# the bias actually depends on likelihood when this 
# function is called 
# and prob. 0.5 of tails
def flip_coin(bias=0.5):
    p1 = bias
    p0 = 1-p1
    return np.random.choice([0,1],p=[p0,p1])   


# used to get samples for flip
def coin_samples(n=1, function=None, bias=0.5):
    sample_points = []
    if function is None:
        return
    else:
        for x in range(n):
            if function==flip_coin:
                sample_points.extend([function(bias)])
            else:        
                sample_points.extend([function()])
    return sample_points


def get_coin_samples(num_samples=10000, bias=0.5):
    samples = coin_samples(num_samples, function=flip_coin, bias=bias)
    return samples

def plot_coin_samples(samples):
    plt = hist(samples)
    plt.show()
    

def hist(samples):
    plt.figure()
    #plt.hist(samples,2,normed=1,facecolor='green', alpha=0.75)
    results, edges = np.histogram(samples, normed=True)
    binWidth = edges[1] - edges[0]
    plt.bar(edges[:-1], results*binWidth, binWidth)
    return plt


################################################## Rejection Sampling #############################################

## Global variables for rejection sampling    
total_samples = 0
rejected_samples = 0
accepted_samples = 0
samples_considred = 0


def rejection_sampling(acc_samples, prior_weights, data):
    result = data_samples(acc_samples, prior_weights, data)
    
    print "Rejection Sampling"
    print "Total Samples: ", total_samples
    print "Rejected Samples: ", rejected_samples
    print "Accepted Samples: ", total_samples - rejected_samples
    print "Samples Considred: ", samples_considred
    
    return result
    

# used to generate n accepted samples for result
def data_samples(acc_samples, prior_weights, data):
    result = init_result()
    #call draw_samples n times
    for x in range(acc_samples):
        draw_Samples(prior_weights, data, result)
    return result
        

def draw_Samples(prior_weights, data, result):
    global total_samples
    global rejected_samples
    global samples_considred  # accepted samples added to the result
    
    #draw hypothesis according to the prior
    ind = np.where(np.random.multinomial(1, prior_weights))[0][0]
    node = nodes[ind]
    total_samples += 1
    
    if all( [is_child(x, node)for x in data] ):
        if flip_coin(likelihood(node, data)):
            samples_considred += 1
            result[node] += 1  
    else: 
        rejected_samples += 1
        draw_Samples(prior_weights, data, result) 
      
    
def init_result(): 
    # dictionary containing final result
    result = collections.OrderedDict()
    for node in nodes:
        result[node] = 0
        #result[nodes.index(node)] = 0
    return result    
        
    
def plot_result(result, title): 
    samp = []
    for key,value in result.items():
        samp.extend([value])
    plt = barchart(samp)
    plt.title(title)
    plt.show()
    
    
def get_prediction(result):
    """
    result is a dictionary with keys as nodes and 
    values as the number of times the respective nodes 
    were predicted
    
    returns the node name e.g., b
    """
    return max(result.iteritems(), key=operator.itemgetter(1))[0]
    

########################################### Metropolis Hastings ###########################################


# this is the function defining the target distribution 
# we want to sample from i.e the posterior
def target(node, data):
    # find the mapping to letters
    node = node_map[node]
    pri = prior(node)
    if all( [is_child(x, node)for x in data] ):
        lik = likelihood(node, data)
    else:
        lik = 0
    return pri*lik
    
    
# symmetric proposal function
# equally likely to propose one number higher or lower
def symm_pfun(x):
    return np.random.choice(range(len(nodes)))
    if flip_coin(0.5):
        return x-1
    else:
        return x+1    
    
    
# symmetric proposal distribution
def symm_pdist(x):    
    return 0.5


def normal_pfun(sigma, mu):
    # return a sample from the normal distribution
    return int(sigma * np.random.randn() + mu)
       
       
# probability of x2 given x1    
def normal_pdist(x, mu, sigma): 
    return ( 1/np.sqrt(2*np.pi*sigma**2) ) *  np.exp(-(x-mu)**2 / 2*sigma**2)
    
    
# uses gamma distribution as the proposal distribution    
def mcmc_symm(num_samples, data):
    z = np.zeros(num_samples)
    a = np.zeros(num_samples)
    z[0] = 0
    sd = 1
    for i in range(2, num_samples):
        
        x = z[i-1]   # old state
        y = symm_pfun(x)   # propose a new state
        #y = normal_pfun(sigma=sd, mu=x)
        
        # accept new y with prob
        if y >= 0 and y < len(nodes):
            rtarget = target(y, data)/target(x, data)         # target ratio
            #rproposal = normal_pdist(y, x, sigma=sd) / normal_pdist(x, y, sigma=sd)   # proposal ratio
            rproposal = symm_pdist(x) / symm_pdist(y)
            p = rtarget*rproposal
        else:
            p = 0    
            
        # generate a u from the uniform distribution     
        u = np.random.uniform(0,1)
        if u < min(p, 1):
            # accept the proposal
            z[i] = y
            a[i] = 1
        else:
            z[i] = x
            a[i] = 0
    
    val = {'a':a, 'z':z}
    
    print "\n Mcmc symmetric distribution"
    print "Samples Accepted: ", sum(a)
    print "Total Samples: ", num_samples
    print z
    np.save("test.npy", z)
    result  = mcmc_result(z)
    return(result)

# uses gamma distribution as the proposal distribution    
def mcmc_epsilon(num_samples, data):
    z = np.zeros(num_samples)
    a = np.zeros(num_samples)
    z[0] = 0
    for i in range(2, num_samples):
        
        x = z[i-1]   # old state
        y = np.random.gamma(shape=1, scale=1)   # propose a new state
        
        # accept new y with prob
        if y >= 0 and y < len(nodes):
            rtarget = target(y, data)/target(x, data)         # target ratio
            #rproposal = symm_pdist(y, x) / symm_pdist(x, y)   # proposal ratio
            rproposal = symm_pdist(x) / symm_pdist(y)
            p = rtarget*rproposal
        else:
            p = 0    
            
        # generate a u from the uniform distribution     
        u = np.random.uniform(0,1)
        if u < min(p, 1):
            # accept the proposal
            z[i] = y
            a[i] = 1
        else:
            z[i] = x
            a[i] = 0
    
    val = {'a':a, 'z':z}
    
    print "\n Mcmc symmetric distribution"
    print "Samples Accepted: ", sum(a)
    print "Total Samples: ", num_samples
    
    np.save("test.npy", z)
    result  = mcmc_result(z)
    return(result)
    
    

def credible_interval(z):
    print "C Interval: ", np.percentile(z, 2.5)
    print "C Interval: ", np.percentile(z, 97.5)
    
def mcmc_result(state_samples):
    """
    state_samples: list of samples drawn through mcmc
    """
    result = init_result()
    for sample in state_samples:
        node = node_map[sample]
        result[node] += 1
    return result    
    
    
################################################################################################################
# Generate from the model and estimate the parameters
def validate_model(prediction, data):
    lik = likelihood(prediction, data)
    pri = prior(prediction)
    
    lik2 = likelihood('a', data)
    pri2 = prior('a')
    
    #print lik
    #print pri
    print "Expectation: ", lik*pri + lik2*pri2
    


################################################################################################################
            

def main():
    np.set_printoptions(threshold=np.nan)
    
    # read data from the csv file to construct
    # nodes, heights, parents and node_maps lists/dictionaries
    read_csv('full_space.csv')
    
    data = [36, 39]    # observed data
    
    # Generating coin samples
    #samples = get_coin_samples(num_samples=10000, bias=0.8)
    #plot_coin_samples(samples=samples)
      
    
    # Rejection Sampling
    prior_weights = cal_prior(nodes)
    result = rejection_sampling(5000, prior_weights, data)
    plot_result(result, "Rejection Sampling")
    
    #prediction = get_prediction(result)
    #validate_model(prediction, data)
    
    
    result = mcmc_symm(num_samples=9000, data=data)
    plot_result(result, "Mcmc Sampling")
    
    
    
    """
    print "Posterior ratio: ", float(result['b']) /result['a']
    bayes_factor = float( likelihood('b', data) ) / likelihood('a', data)
    print "Bayes factor: ", bayes_factor
    prior = cal_prior(nodes)
    prior_odds = float(prior[0]) / prior[2]
    print "Prior Odds: ", prior_odds
    print "posterior odds: ", float(bayes_factor)*prior_odds
    """
    
       
if __name__ == "__main__": main()
