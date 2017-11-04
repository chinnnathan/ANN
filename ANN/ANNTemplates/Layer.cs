using System;
using System.Collections.Generic;
using System.Diagnostics;
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

        public string Label { get; set; }
        public int Index { get; set; }

        public void WeightMatrixOperation()
        {
            try
            {

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
}
