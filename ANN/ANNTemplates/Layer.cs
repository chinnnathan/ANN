using System;
using System.Collections.Generic;
using System.Diagnostics;
using ANN.Utils;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    public class Layer : List<Neuron>
    {
        // each layer could go to n other layers
        public Layer PreviousLayer;
        public Layer NextLayer;

        private double[,] _weights;
        private double[,] _values;
        //private double[,] _outputs;
        private double[][] _outputs;
        private double[][] _gradients;

        public string Label { get; set; }
        public int Index { get; set; }
        public LayerType Type { get; set; }
        public double[] Predictions { get { return this.Select(x=>x.Prediction).ToArray(); } }
        public double[] OutputWeights { get { return this.SelectMany(x => x.InterOut.Select(y => y.Weight)).ToArray(); } }

        public void FeedForward()
        {
            try
            {
                if (Type == LayerType.Output)
                {
                    List<double> v = this.Select(x => x.Input).ToList();
                    Parallel.ForEach(this, neuron =>
                    {
                        neuron.Prediction = (double) neuron.RunActivation(neuron.Input, v.ToArray());
                        neuron.BpropValue = (double)neuron.RunGradient(neuron.Expected, neuron.Prediction);
                        neuron.Error = GradientFunctions.SumSquaredError(neuron.Expected, neuron.Prediction);
                    });
                    Parallel.ForEach(this, neuron =>
                    {
                        Parallel.ForEach(neuron.InterIn, input =>
                        {
                            input.Error = neuron.BpropValue;
                        });
                    });
                }
                else
                {
                    Parallel.ForEach(this, neuron =>
                    {
                        neuron.Prediction = (double)neuron.RunActivation(neuron.Input);
                        neuron.BpropValue = (double)neuron.RunGradient(neuron.Input);
                    });

                    _values = AdvancedMath.ConvertToMatrix(
                        this.Select(x => x.Prediction).ToArray(),
                        -1, 1);

                    _weights = AdvancedMath.ConvertToMatrix(
                        this.SelectMany(x => x.InterOut.Select(w => w.Weight)).ToArray(),
                        this[0].InterOut.Count, Count);

                    _outputs = AdvancedMath.JMultiplyMatrix(_weights, _values);

                    Parallel.ForEach(this, neuron =>
                    {
                        Parallel.ForEach(neuron.InterOut, inter =>
                        {
                            inter.DValue = neuron.BpropValue;
                            inter.FValue = neuron.Prediction;
                            inter.IValue = neuron.Input;
                        });
                    });

                    Parallel.For(0, _outputs.Count(), i =>
                    {
                        Parallel.For(0, _outputs[i].Count(), j =>
                        {
                            NextLayer[i].Input = _outputs[i][j];
                        });
                    });
                }
            }
            catch (Exception e)
            {
                Debug.WriteLine("Exception: {0}", e.ToString());
            }
        }

        public void BackPropagate()
        {
            try
            {
                // All Derivative functions are ran in Feedforward for ease
                // Error still needed to be found
                // Update the weights based on backpropagation value
                if (Type == LayerType.Output)
                {
                    //Delta = alpha * (tk-yk) * f'(y_ink)
                    //      = LearningRate * Error Value * Derivative Value
                    Parallel.For(0, Count, i =>
                    {
                        Parallel.For(0, this[i].InterIn.Count, j =>
                        {
                            this[i].InterIn[j].Update += this[i].LearningRate *
                                this[i].BpropValue * this[i].InterIn[j].DValue;
                        });
                    });
                }

                else
                {
                    //Delta = Sigma((tk-yk)(f'(yin_k) * wjk)
                    //      where z = this layer, y = next layer
                    //Gradient = alpha * Delta * f'(z_inj) * xi
                    //      = LearningRate * Sum of( Error Value * weight)
                    //          * Derivative Value * Value input
                    Parallel.ForEach(this, neuron =>
                    {
                        neuron.BpropValue = 0;
                        neuron.Error = 0;
                        Parallel.ForEach(neuron.InterOut, output =>
                        {
                            neuron.Error += output.Error;
                            neuron.BpropValue += output.Error * output.DValue * output.Weight;
                        });
                    });

                    Parallel.ForEach(this, neuron =>
                    {
                        Parallel.ForEach(neuron.InterIn, input =>
                        {
                            input.Error = neuron.Error;
                        });
                    });

                    Parallel.For(0, Count, i =>
                    {
                        Parallel.For(0, this[i].InterIn.Count, j =>
                        {
                            this[i].InterIn[j].Update += this[i].LearningRate *
                                this[i].BpropValue * this[i].InterIn[j].DValue *
                                this[i].InterIn[j].IValue;
                        });
                    });
                }
                
                
                
            }
            catch (Exception e)
            {
                Debug.WriteLine("Exception: {0}", e.ToString());
            }
        }


        public void StandardConnectNeurons(double min, double max)
        {
            try
            {
                foreach (var neuron in this)
                {
                    if (Type == LayerType.Input)
                    {
                        NeuronInterconnect icIn = new NeuronInterconnect()
                        {
                            Weight = AdvancedMath.GetRandomRange(min, max),
                        };
                        neuron.InterIn.Add(icIn);
                    }

                    foreach (var nextNeuron in NextLayer)
                    {
                        NeuronInterconnect icl1 = new NeuronInterconnect()
                        {
                            Weight = AdvancedMath.GetRandomRange(min,max),
                        }; // initialize layer1
                        NeuronInterconnect icl2 = icl1; // reference layer1 for use in layer2
                        neuron.InterOut.Add(icl1);
                        nextNeuron.InterIn.Add(icl2);
                        neuron.AfterNeurons.Add(nextNeuron);
                        nextNeuron.PriorNeurons.Add(neuron);
                    }
                }
            }
            catch (Exception e)
            {
                Debug.WriteLine("Exception: {0}", e.ToString());
            }
        }
    }

    public enum LayerType
    {
        Input,
        Hidden,
        Output,
    }
}
