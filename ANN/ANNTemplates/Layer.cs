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
        private double[,] _outputs;

        public string Label { get; set; }
        public int Index { get; set; }
        public LayerType Type { get; set; }

        public void FeedForward()
        {
            try
            {
                _weights = AdvancedMath.ConvertToMatrix(
                    this.SelectMany(x => x.InterOut.Select(w => w.Weight)).ToArray(), 
                    this[0].InterOut.Count, Count);

                _values = AdvancedMath.ConvertToMatrix(
                    this.Select(x => x.Prediction).ToArray(),
                    -1, 1);

                if (Type == LayerType.Output)
                {
                    List<double> v = _values.Cast<double>().ToList();

                    _outputs = new double[_values.Length, 1];
                    Parallel.For(0, _outputs.Length, i =>
                    {
                        _outputs[i, 0] = (double)this[i].RunActivation(_values[i,0], v.ToArray());
                    });
                }
                else if (NextLayer.Type == LayerType.Output)
                {
                    _outputs = new double[_values.Length, 1];
                    Parallel.For(0, _values.Length, i =>
                    {
                        _outputs[i, 0] = this[i].Prediction;
                    });
                }
                else
                {
                    _outputs = AdvancedMath.MultiplyMatrix(_weights, _values);
                    Parallel.For(0, _outputs.Length, i =>
                    {
                        _outputs[i, 0] = (double)this[i].RunActivation(_outputs[i, 0]);
                        NextLayer[i].Input = _outputs[i, 0];
                    });
                }
            }
            catch (Exception e)
            {
                Debug.WriteLine("Exception: {0}", e.ToString());
            }
        }

        public void StandardConnectNeurons()
        {
            try
            {
                foreach (var neuron in this)
                {
                    foreach (var nextNeuron in NextLayer)
                    {
                        NeuronInterconnect icl1 = new NeuronInterconnect(); // initialize layer1
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
