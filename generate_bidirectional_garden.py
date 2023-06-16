import sys
import random
import copy
from num2words import num2words

# Matéo: count according to colour adjectives only


fruit_filename = sys.argv[1]

adj_filename = sys.argv[2]

max_quantity = int(sys.argv[3])

num_items = int(sys.argv[4])

outfilename = sys.argv[5]


def generate_quantified_statement(quantity, adj, fruit):
    if quantity == 1:
        fruit = singular_of[fruit]
    return "you have " + num2words(quantity) + " " + adj + " " + fruit


def generate_randomized_context(adj, quantity, n_merge):
    list_of_statements = []
    # random value below n_items statements with target adjectives
    values = 0
    for i in range(n_merge):
        # avoid 0
        if quantity - values == 0:
            # repetition risk
            n_merge -= n_merge - i
            break
        elif i != n_merge - 1:
            _v = random.randint(1, quantity - values)
            values += _v
        else:
            _v = quantity - values
        fruit = random.choice(fruits)
        list_of_statements.append(generate_quantified_statement(_v, adj, fruit))

    for i in range(num_items - n_merge):
        _v = random.randint(1, max_quantity)
        fruit = random.choice(fruits)
        # exclude the target adjective
        remaining_adjs = copy.deepcopy(adjs)
        remaining_adjs.remove(adj)
        _a = random.choice(remaining_adjs)
        list_of_statements.append(generate_quantified_statement(_v, _a, fruit))
    random.shuffle(list_of_statements)
    return list_of_statements


if __name__ == "__main__":
    fruits = []
    singular_of = dict()

    ifile = open(fruit_filename, "r")
    lines = ifile.readlines()
    for line in lines:
        (singular, plural) = line.strip().split("\t")
        fruits.append(plural)
        singular_of[plural] = singular
    ifile.close()

    with open(adj_filename, "r") as f:
        adjs = f.readlines()
    adjs = [x.strip() for x in adjs]

    # for every number up to max_quantity, generate a statement for every adjective
    # and every fruit
    for indicator in ["the other network", "you", "you and the other network"]:
        for adj in adjs:
            question = "how many " + adj + " items do " + indicator + " have?"
            for quantity in range(1, max_quantity + 1):
                # have up to n_items generated statements with target adjectives while the rest use other adjectives
                for n_merge in range(1, num_items + 1):
                    # generate randomized context
                    # only distractors as context
                    for _qtt in [
                        quantity if indicator == "you" else 0,
                        random.randint(1, quantity),
                    ]:
                        list_context_llm1 = generate_randomized_context(
                            adj, _qtt, n_merge
                        )
                        list_context_llm2 = generate_randomized_context(
                            adj, quantity - _qtt, n_merge
                        )
                        data_point = (
                            ", ".join(list_context_llm1)
                            + " "
                            + question
                            + " \t"
                            + ", ".join(list_context_llm2)
                            + " \t"
                            + str(quantity)
                            + " \t"
                            + adj
                            + "\n"
                        )
                        print(data_point)

                        # output to file
                        with open(outfilename, "a") as f:
                            f.write(data_point)

# randomize the order of the outfile
with open(outfilename, "r") as f:
    lines = f.readlines()
random.shuffle(lines)
with open(outfilename, "w") as f:
    f.writelines(lines)

# take 10% of the outfile as test set, 10% as val set, and the rest as train set
with open(outfilename, "r") as f:
    lines = f.readlines()
n_lines = len(lines)
n_test = int(n_lines * 0.1)
n_val = int(n_lines * 0.1)
n_train = n_lines - n_test - n_val
with open(outfilename[:-4] + "_test.txt", "w") as f:
    f.writelines(lines[:n_test])
with open(outfilename[:-4] + "_valid.txt", "w") as f:
    f.writelines(lines[n_test : n_test + n_val])
with open(outfilename[:-4] + "_train.txt", "w") as f:
    f.writelines(lines[n_test + n_val :])
