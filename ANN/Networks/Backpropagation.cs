using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANN.ANNTemplates;
using System.Diagnostics;

namespace ANN.Networks
{
    public class Backpropagation : Network
    {

        public Backpropagation() { }

        public Backpropagation(int outputnodes, int hiddenlayers=1)
        {
            Layer input = new Layer()
            {
                Index = 0,
                Label = "Input",
            };
            Layer output = new Layer()
            {
                Index = hiddenlayers + 1,
                Label = "Output",
            };
            Add(input);
            Add(output);

            for(int i=0; i<hiddenlayers; i++)
            {
                Add(new Layer()
                {   Index = i + 1,
                    Label = string.Format("HiddenLayer_{0}", i),
                });
            }
        }

        
        
    }
}
