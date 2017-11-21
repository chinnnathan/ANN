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

namespace ANNGUI.UserIO
{
    /// <summary>
    /// Interaction logic for VariableSet.xaml
    /// </summary>
    public partial class VariableSet : UserControl
    {
        private List<string> _variables = new List<string>() { "Temporary", "Debugging" };

        public VariableSet()
        {
            InitializeComponent();
            int i = 0;
            foreach (var variable in _variables)
            {
                ListViewItem lvi = new ListViewItem();
                listView.Items.Add(variable);
                listView.Items.Add(i);
                i++;
            }
        }

    }
}
