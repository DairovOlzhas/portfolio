using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Calculator
{
    class Calculator
    {
        public static double fn = new int(), sn = new int();
        public static bool b = true, e = true;
        public static string oper = "";
        public static double memory;

        public static string Oper_Change(string n, string oper)
        {
            if (n == "") return "";
            double res = double.Parse(n);
            double sn = double.Parse(n);
            if (oper == "ln")
                res = Math.Log(sn);

            if (oper == "cos")
                res = Math.Cos(sn * Math.PI / 180);

            if (oper == "sin")
                res = Math.Sin(sn * Math.PI / 180);

            if (oper == "tan")
                res = Math.Tan(sn * Math.PI / 180);

            if (oper == "√")
                res = Math.Sqrt(sn);

            if (oper == "x!")
            {
                res = 1;
                for (int i = 1; i <= sn; i++)
                    res *= i;
            }

            if (oper == "EXP")
                res = Math.Pow(Math.E, sn);

            if (oper == "1/x")
                res = 1 / sn;

            if (oper == "%")
                res = fn * sn / 100;

            return res + "";
        }

        public static string Calc()
        {
            double res = new double();

            if (oper == "+")
                res = (fn + sn);

            if (oper == "÷")
                res = (fn / sn);

            if (oper == "*")
                res = (fn * sn);

            if (oper == "-")
                res = fn - sn;

            if (oper == "log")
                res = Math.Log(fn, sn);      

            if (oper == "x^y")
                res = Math.Pow(fn, sn);            

            return res + "";
        }

        public static string Oper(string textBox1, string btn)
        {
            if (textBox1 == "") return "";
            if (e)
            {
                fn = new double();
                sn = new double();
                e = false;
            }
            if (fn == new double())
            {
                fn = double.Parse(textBox1);
                b = true;
            }
            else if (!b)
            {
                sn = double.Parse(textBox1);

                textBox1 = Calc();

                fn = double.Parse(textBox1);

                b = true;
            }

            oper = btn;

            return textBox1;
        }

        public static string Equal(string textBox1)
        {
            if (oper == "")
                return textBox1;

            if ((sn == new double() || !e) && fn != new double())
                sn = double.Parse(textBox1);
            if (e && fn != double.Parse(textBox1))
                fn = double.Parse(textBox1);
            string res = Calc();
            
            fn = double.Parse(res);
            b = true;
            e = true;
            return res;
        }

        public static void Clear(string btn)
        {
            if(btn == "C")
            {
                sn = new double();
                fn = new double();
                oper = "";
            }
            else
            {
                if(e && sn != new double())
                    fn = new double();
            }
            
        }

        public static string Memory( string Text, string btn)
        {
            if(btn == "MS")
                memory = double.Parse(Text);
            
            if(btn == "MR")
                return memory + "";
            
            if(btn == "MC")
                memory = new double();
            
            if(btn == "M+")            
                memory += double.Parse(Text);
            
            if (btn == "M-")            
                memory -= double.Parse(Text);
            return "$";
        }
    }
}
