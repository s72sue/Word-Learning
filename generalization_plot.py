import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys


if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file containing data to generate the plot"
    exit(0)




def pg_barplot(pg_list, title):
    
    ind = np.arange(len(pg_list))
    width = 1                       
    plt.figure(figsize=(5, 6), facecolor='white')
    plt.bar(ind, pg_list, width, color='grey') #, yerr=menStd)
    plt.xlabel('Categories', fontsize=11)
    plt.ylabel('Probability of generalization', fontsize=12)
    plt.title(title)
    plt.xticks(ind + width/2., ['sub', 'basic', 'super'])
    #plt.ylim(0,1)
    #plt.show()
    plt.savefig('./generalization_plots/' + title, fontsize=12)
    plt.close()
    return plt
    
    
    
data = pickle.load(open(fname, 'rb'))

vegetable_1sub = data['vegetable_1sub']
vegetable_3sub = data['vegetable_3sub']
vegetable_3basic = data['vegetable_3basic']
vegetable_3sup = data['vegetable_3sup']

vehicle_1sub = data['vehicle_1sub']
vehicle_3sub = data['vehicle_3sub']
vehicle_3basic = data['vehicle_3basic']
vehicle_3sup = data['vehicle_3sup']

animal_1sub = data['animal_1sub']
animal_3sub = data['animal_3sub']
animal_3basic = data['animal_3basic']
animal_3sup = data['animal_3sup']


# Average the data from three categories
sub1 = [sum(x)/3.0 for x in zip(vegetable_1sub, vehicle_1sub, animal_1sub)]
sub3 = [sum(x)/3.0 for x in zip(vegetable_3sub, vehicle_3sub, animal_3sub)]
basic3 = [sum(x)/3.0 for x in zip(vegetable_3basic, vehicle_3basic, animal_3basic)]
sup3 = [sum(x)/3.0 for x in zip(vegetable_3sup, vehicle_3sup, animal_3sup)]


pg_barplot(sub1, '1 sub')
pg_barplot(sub3, '3 sub')
pg_barplot(basic3, '3 basic')
pg_barplot(sup3, '3 sup')















