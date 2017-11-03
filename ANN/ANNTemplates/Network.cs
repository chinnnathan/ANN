using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    public class Network : List<Layer>
    {

        public Network() { }
        public Network(int layerCount, int neuronCount)
        {
            for(int l=0;l<layerCount;l++)
            {
                Layer layer = new Layer();
                for(int n=0;n<neuronCount;n++)
                {
                    Neuron neuron = new Neuron();
                    layer.Add(neuron);
                }
                this.Add(layer);
            }
        }
    }
}
