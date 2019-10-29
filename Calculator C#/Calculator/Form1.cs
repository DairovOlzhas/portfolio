using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Calculator
{
    public partial class Form1 : Form
    {
        
        public Form1()
        {
            InitializeComponent();
        }

        private void numbers_click(object sender, EventArgs e)
        {
            Button btn = sender as Button;
            if (btn.Text == "," && textBox1.Text == "")
                textBox1.Text = "0,";
            else if ((Calculator.b || textBox1.Text == "0") && btn.Text != ",")
            {
                textBox1.Text = btn.Text;
                Calculator.b = false;
            }
            else
                textBox1.Text += btn.Text;
        }

        private void Oper_Value_click(object sender, EventArgs e)
        {
            Button btn = sender as Button;

            if (btn.Text == "±" && textBox1.Text != "" && textBox1.Text != "0")
            {
                if (textBox1.Text[0] == '-')
                    textBox1.Text = textBox1.Text.Remove(0, 1);
                else
                    textBox1.Text = "-" + textBox1.Text;
            }
            else if (btn.Text == "e")
            {
                textBox1.Text = Math.E.ToString();
            }
            else if (btn.Text == "π")
            {
                textBox1.Text = Math.PI.ToString();
            }
            else
                textBox1.Text = Calculator.Oper_Change(textBox1.Text, btn.Text);
        }

        private void operation_click(object sender, EventArgs e)
        {
            Button btn = sender as Button;
            textBox1.Text = Calculator.Oper(textBox1.Text, btn.Text);            
        }

        private void backspace(object sender, EventArgs e)
        {
            if(textBox1.Text.Length > 0)
                textBox1.Text = textBox1.Text.Remove(textBox1.Text.Length - 1);
            if (textBox1.Text == "" || textBox1.Text == "-")
                textBox1.Text = "0";
        }

        private void button_C(object sender, EventArgs e)
        {
            Button btn = sender as Button;

            Calculator.Clear(btn.Text);
            textBox1.Text = "0";
        }

        private void equal_click(object sender, EventArgs e)
        {
            textBox1.Text = Calculator.Equal(textBox1.Text);            
        }

        private void memory_click(object sender, EventArgs e)
        {
            Button btn = sender as Button;

            if(Calculator.Memory(textBox1.Text, btn.Text) != "$")            
                textBox1.Text = Calculator.Memory(textBox1.Text, btn.Text);
            
        }




    }
}
