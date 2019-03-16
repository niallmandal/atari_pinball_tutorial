import gym
import numpy as np
import cv2
import neat
import pickle

import random
import noise

env = gym.make('VideoPinball-v0')
imgarray=[]

def eval_genomes(genomes, config):
    for genome_id,genome in genomes:
        ob = env.reset()
        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        fitness_current = 0
        frame = 0
        counter = 0

        done = False

        while not done:
            frame += 1

            factor = 0.5

            ob = np.uint8(noise.noisy(ob,factor))
            ob = cv2.resize(ob, (inx,iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)

            imgarray = np.ndarray.flatten(ob)
            nnOutput = net.activate(imgarray)

            numerical_input = nnOutput.index(max(nnOutput))
            ob, rew, done, info = env.step(numerical_input)

            fitness_current += rew

            if rew>0:
                counter = 0
            else:
                counter += 1

            env.render()
            if done or counter == 250:
                done = True
                print(genome_id,fitness_current)
            genome.fitness = fitness_current

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward.txt')

p = neat.Population(config)

p.add_reporter(neat.Checkpointer(10))
winner = p.run(eval_genomes)

with open('winner_pinball1.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
