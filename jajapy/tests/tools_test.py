import unittest
from ..base.tools import *

class ToolsTestclass(unittest.TestCase):

	def test_resolveRandom(var):
		probabilities = [0.3,0.6,0.1]
		nb_trials = 100000
		results = [0.0,0.0,0.0]
		for i in range(nb_trials):
			results[resolveRandom(probabilities)] += 1/nb_trials
		for i in range(len(probabilities)):
			var.assertAlmostEqual(probabilities[i],results[i],delta=3)
	
	def test_randomProbabilities(var):
		var.assertRaises(ValueError, randomProbabilities, 0)
		var.assertRaises(TypeError, randomProbabilities, 0.2)
		for length in range(1, 6):
			p = randomProbabilities(length)
			var.assertEqual(len(p),length)
			var.assertEqual(sum(p),1.0)
			for i in range(length):
				var.assertGreaterEqual(p[i],0.0)


if __name__ == "__main__":
	unittest.main()