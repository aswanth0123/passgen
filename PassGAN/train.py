import os, sys
sys.path.append(os.getcwd())

import time
import pickle
import argparse
import numpy as np
import tensorflow as tf

import utils
import tflib as lib
import tflib.ops.l2
import tflib.ops.con1
import tflib.plot
import mod1

'''

$ python train.py -o "pretrained"

'''

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-i',
                        default='data/text.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/text.txt)')

    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    parser.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=24,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')
    
    parser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    
    parser.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')
    
    return parser.parse_args()

args = parse_args()

lines, charmap, inv_charmap = utils.load_dataset(
    path=args.training_data,
    max_length=args.seq_length)

print(args)
# Pickle to avoid encoding errors with json
with open(os.path.join(args.output_dir, 'charmap.pickle'), 'wb') as f:
    pickle.dump(charmap, f)

with open(os.path.join(args.output_dir, 'charmap_inv.pickle'), 'wb') as f:
    pickle.dump(inv_charmap, f)
    
print("Number of unique characters in dataset: {}".format(len(charmap)))

real_inputs_discrete = tf.compat.v1.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))

fake_inputs = mod1.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

disc_real = mod1.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap))
disc_fake = mod1.Discriminator(fake_inputs, args.seq_length, args.layer_dim, len(charmap))

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN lipschitz-penalty
alpha = tf.random.uniform(
    shape=[args.batch_size,1,1],
    minval=0.,
    maxval=1.
)

differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
gradients = tf.gradients(mod1.Discriminator(interpolates, args.seq_length, args.layer_dim, len(charmap)), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += args.lamb * gradient_penalty

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

gen_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-args.batch_size+1, args.batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+args.batch_size]],
                dtype='int32'
            )
            
# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[10*args.batch_size:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[:10*args.batch_size], tokenize=False) for i in range(4)]
for i in range(4):
    print("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]


# TensorFlow Session
with tf.compat.v1.Session() as session:

    # Time stamp
    localtime = time.asctime( time.localtime(time.time()) )
    print("Starting TensorFlow session...")
    print("Local current time :", localtime)
    
    # Start TensorFlow session...
    session.run(tf.compat.v1.global_variables_initializer())

    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    gen = inf_train_gen()

    for iteration in range(args.iters + 1):
        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        for i in range(args.critic_iters):
            _data = next(gen)
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete:_data}
            )

        lib.plot.output_dir = args.output_dir
        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)
        # Output to text file after every 100 samples
        if iteration % 100 == 0 and iteration > 0:

            samples = []
            for i in range(10):
                samples.extend(generate_samples())

            for i in range(4):
                lm = utils.NgramLanguageModel(i+1, samples, tokenize=False)
                lib.plot.plot('js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))
            print('arg',args.output_dir)
            with open(os.path.join(args.output_dir, 'samples_{}.txt').format(iteration), "w", encoding='utf-8') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")

        if iteration % args.save_every == 0 and iteration > 0:
            model_saver = tf.compat.v1.train.Saver()
            model_saver.save(session, os.path.join(args.output_dir, 'checkpoints', 'checkpoint_{}.ckpt').format(iteration))
            print("{} / {} ({}%)".format(iteration, args.iters, iteration/args.iters*100.0 ))

        if iteration == args.iters:
            print("...Training done.")
        
        if iteration % 100 == 0:
            lib.plot.flush()

        lib.plot.tick()
        
# Time stamp
localtime = time.asctime( time.localtime(time.time()) )
print("Ending TensorFlow session.")
print("Local current time :", localtime)
