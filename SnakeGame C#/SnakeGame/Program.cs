using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SnakeGame
{
    class Program
    {
        public static Game game = new Game();
        
        static void GameProcess()
        {
            Console.Clear();
            Thread moving = new Thread(Game.GameProcess);
            moving.Start();
           

            while (Game.check && Game.btn.Key != ConsoleKey.Escape)
            {                
                Game.btn = Console.ReadKey();
                if (Game.btn.Key == ConsoleKey.RightArrow && Game.dy != 0){ Game.dx = 1;  Game.dy = 0; }
                if (Game.btn.Key == ConsoleKey.LeftArrow && Game.dy != 0) { Game.dx = -1; Game.dy = 0; }
                if (Game.btn.Key == ConsoleKey.UpArrow && Game.dx != 0)   { Game.dx = 0;  Game.dy = -1;}
                if (Game.btn.Key == ConsoleKey.DownArrow && Game.dx != 0) { Game.dx = 0;  Game.dy = 1; }
            }

            while (Game.btn.Key != ConsoleKey.Enter && Game.btn.Key != ConsoleKey.Escape)
            {
                Console.SetCursorPosition(0, 19);
                Game.btn = Console.ReadKey();
                Console.SetCursorPosition(0, 19);
                Console.WriteLine("    ");
            }
            moving.Join();

            Console.Clear();
            
        }

        static void ShowUsers(int pos)
        {
            DirectoryInfo users = new DirectoryInfo("Users");
            int i = 0;
            foreach(DirectoryInfo d in users.GetDirectories())
            {
                if (i == pos)
                {
                    Console.ForegroundColor = ConsoleColor.Black;
                    Console.BackgroundColor = ConsoleColor.White;
                }
                Console.WriteLine(d.Name);

                Console.ForegroundColor = ConsoleColor.White;
                Console.BackgroundColor = ConsoleColor.Black;
                i++;
            }
        }
        static void UsersLogic()
        {
            int pos = 0;
            ConsoleKeyInfo btn = new ConsoleKeyInfo();
            DirectoryInfo users = new DirectoryInfo("Users");

            while (btn.Key != ConsoleKey.Escape)
            {
                if (pos > users.GetDirectories().Length - 1)
                    pos = 0;
                if (pos < 0)
                    pos = users.GetDirectories().Length - 1;

                Console.Clear();

                ShowUsers(pos);

                btn = Console.ReadKey();
                if (btn.Key == ConsoleKey.UpArrow)
                    pos--;
                if (btn.Key == ConsoleKey.DownArrow)
                    pos++;
                if (btn.Key == ConsoleKey.Enter && users.GetDirectories().Length > 0)
                {
                    Game.btn = new ConsoleKeyInfo();
                    Game.check = true;
                    Game.name = users.GetDirectories()[pos].Name;
                    Game.ReadDataGame(Game.name);
                    GameProcess();

                    break;
                }
                if (btn.Key == ConsoleKey.Delete && users.GetDirectories().Length > 0)
                {
                    foreach(FileInfo f in users.GetDirectories()[pos].GetFiles())
                    {
                        f.Delete();
                    }
                    users.GetDirectories()[pos].Delete();
                }
            }
        }

        static void ShowMenu(int pos)
        {
            Console.Clear();
            string[] buttons = { "Start New Game", "Resume", "Load Game", "Save", "Exit" };
            int i = 0;
            foreach(string b in buttons)
            {
                Console.BackgroundColor = (i == pos) ? ConsoleColor.White : ConsoleColor.Black;
                Console.ForegroundColor = (i == pos) ? ConsoleColor.Black : ConsoleColor.White;

                Console.WriteLine(b);

                Console.BackgroundColor = ConsoleColor.Black;

                i++;
            }
        }
        static void MenuLogic()
        {
            int pos = 0;
            bool exit = false;
            ConsoleKeyInfo btn = new ConsoleKeyInfo();

            while (!exit)
            {
                if (pos > 4) pos = 0;
                if (pos < 0) pos = 4;
                ShowMenu(pos);

                btn = Console.ReadKey();

                if (btn.Key == ConsoleKey.UpArrow)
                    pos--;
                if (btn.Key == ConsoleKey.DownArrow)
                    pos++;
                if (btn.Key == ConsoleKey.Enter)
                {
                    switch (pos)
                    {
                        case 0:
                            Game.NewGame();
                            GameProcess();
                            break;
                        case 1:
                            Game.btn = new ConsoleKeyInfo();
                            Game.check = true;
                            GameProcess();
                            break;
                        case 2:
                            UsersLogic();
                            break;
                        case 3:
                            Game.SaveDataGame(true);
                            break;
                        case 4:
                            exit = true;
                            break;
                    }
                }                    
                
            }
        }

        static void Main(string[] args)
        {
            Game.NewGame();
            int x = Game.sizex;
            int y = Game.sizey;

            Console.CursorVisible = false;
            Console.SetWindowSize(x, y);
            //          Game.Draw();
            //            Console.ReadKey();
            //GameProcess();
            MenuLogic();

        }
    }
}
