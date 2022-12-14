import sys
import random
import copy
from num2words import num2words

# usage:
# python generate_mindless_garden.py fruitfile max_quantity num_distractors

# where:

# - fruitfile is a file with fruit/veggie names, one per line in
# singularTABplural format

# - max_quantity is the top quantity for which a statement will be generated

# - num_distractors is the number of irrelevant statements that will also be generated

# e.g.:
# python generate_mindless_garden.py fruits.txt 20 2

# will generate statements of the form "you have N Xs" with N ranging
# from 1 to 20, where each target is accompanied by 2 distractor
# statements

# each output line has num_distractors+1 statements in random
# comma+space-delimited order, one of them being the target, followed
# by tab, followed by the question, followed by tab, followed by the
# answer, followed by tab, followed by the target fruit/name (which
# could contain a space), followed by tab, followed by the index of
# the statement concerning the target in the list of comma-delimited
# statements (counting from 0)

# a line is generated for each possible NxX combination: each line has
# a distinct target statement, and the distractor statements always
# pertain to distinct fruits/veggies; however, distractor statements
# can be repeated across lines

# while the target statement is randomized with respect to the
# distractors, the output lines are ordered, by target statement,
# based on the order of fruits/veggies in the input file, then by
# increasing quantities



fruit_filename = sys.argv[1]

max_quantity = int(sys.argv[2])

num_distractors =  int(sys.argv[3])

def generate_quantified_statement(quantity,fruit):
    if (quantity == 1):
        fruit = singular_of[fruit]
    return "you have " + num2words(quantity) + " " + fruit

fruits = []
singular_of = dict()

ifile = open(fruit_filename,'r')
lines = ifile.readlines()
for line in lines:
    (singular,plural) = line.strip().split('\t')
    fruits.append(plural)
    singular_of[plural] = singular
ifile.close()

stimulus_list = []

for target_fruit in fruits:
    other_fruits = copy.deepcopy(fruits)
    other_fruits.remove(target_fruit)
    question = "how many " + target_fruit + " does the other network have?"
    for quantity in range(1,max_quantity+1):
        list_of_statements = []
        target_statement = generate_quantified_statement(quantity,target_fruit)
        list_of_statements.append(target_statement)
        distractor_fruits = random.sample(other_fruits,num_distractors)
        distractor_quantities = random.choices(range(1,max_quantity+1),k=num_distractors)
        for i in (range(num_distractors)):
            list_of_statements.append(generate_quantified_statement(distractor_quantities[i],distractor_fruits[i]))
        random.shuffle(list_of_statements)
        target_statement_index = list_of_statements.index(target_statement)
        inflected_fruit = target_fruit 
        if quantity == 1:
            inflected_fruit = singular_of[target_fruit]
        print(', '.join(list_of_statements) + '\t' + question +  '\t' + num2words(quantity) + '\t' + inflected_fruit +  '\t' + str(target_statement_index))
