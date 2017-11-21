using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Wpf.CartesianChart.MaterialCards;
using ANN;
using ANN.Utils;

namespace ANNGUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        ANN.ANN ann = new ANN.ANN();

        private List<string> args = new List<string>();

        public MainWindow()
        {
            InitializeComponent();
            OutputAddLine("Working");
            OutputAddLine("Even for multiple values");
            OutputAddLine("This is fantastic");

            foreach (var arg in ann.GetVariableNames())
            {
                LVData lvd = new LVData(arg.Item1, arg.Item2);
                variableSet.Items.Add(lvd);
            }
        }

        private void TrainNetworkBtn_Click(object sender, RoutedEventArgs e)
        {
            string[] arg = new string[] { "-Epochs", "10" };
            ann.RunProgram(arg);
        }

        private void OutputAddLine(string text)
        {
            OutputText.AppendText(text);
            OutputText.AppendText("\u2028"); //Linebreak, not paragraph
            OutputText.ScrollToEnd();
        }

        private string _lastVal = "";
        private bool _update = false;

        private void TextBox_LostFocus(object sender, RoutedEventArgs e)
        {
            string val = ((TextBox)sender).Text;
            if (val != _lastVal)
            {
                _update = true;
                _lastVal = val;
            }
        }

        private void TextBox_GotFocus(object sender, RoutedEventArgs e)
        {
            _lastVal = ((TextBox)sender).Text;
            _update = false;
        }

        private void variableSet_LostFocus(object sender, RoutedEventArgs e)
        {
            if (_update)
            {
                var cd = ((DataGrid)sender).Items;
                foreach (LVData item in cd)
                {
                    args.Add(item.Name);
                    args.Add(string.Format("{0}", item.Value));
                }
                _update = false;
            }
            
        }
    }

    public struct LVData
    {
        private string _name;
        private dynamic _value;

        public string Name { get { return _name; } set { _name = value; } }
        public dynamic Value { get { return _value; } set { _name = value; } }

        public LVData(string varName, dynamic value)
        {
            _name = varName;
            _value = value;
        }
    }

}
