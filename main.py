from PIL import Image, ImageDraw
from ast import arg
import argparse
import numpy as np
import scipy.signal as sp
from rich.progress import track


# obsluga argumentow
parser = argparse.ArgumentParser()

parser.add_argument('-n', '--bok_siatki', default=500)
parser.add_argument('-j', '--wartosc_J', default=10)
parser.add_argument('-b', '--wartosc_beta', default=1)
parser.add_argument('-B', '--wartosc_B', default=1)
parser.add_argument('-N', '--liczba_krokow', default=10)
parser.add_argument('-g', '--gestosc_dodatnich_spinow', default=0.5)
parser.add_argument('-p', '--prefix_nazw_rysonkow', default=[])
parser.add_argument('-a', '--nazwa_pliku_z_animacja_bez_rozszerzenia', default=[])
parser.add_argument('-m', '--nazwa_pliku_z_magnetyzacja', default=[])

args = parser.parse_args()


class Symulacja:
    def __init__(self):
        self.rozmiar = int(args.bok_siatki)
        self.J = int(args.wartosc_J)
        self.beta = int(args.wartosc_beta)
        self.B = int(args.wartosc_B)
        self.N = int(args.liczba_krokow)
        self.prefix = args.prefix_nazw_rysonkow
        self.gestosc = float(args.gestosc_dodatnich_spinow)
        self.animacja = args.nazwa_pliku_z_animacja_bez_rozszerzenia
        self.plik_m = args.nazwa_pliku_z_magnetyzacja

        # inicjacja siatki w zaleznosci od gestosci (w przypadku nieparzystej liczby spinow wiecej jest jedynek)
        self.spiny = np.concatenate((-np.ones(int(np.floor(self.rozmiar**2 * (1-self.gestosc)))), np.ones(int(np.ceil(self.rozmiar**2 * self.gestosc)))))
        np.random.shuffle(self.spiny)
        self.spiny = self.spiny.reshape(self.rozmiar, self.rozmiar)
    def obrazek(self):
        img = Image.new('RGB', (2000, 2000), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        krok = 2000/self.rozmiar
        for j in range(self.rozmiar):
            for i in range(self.rozmiar):
                if self.spiny[i, j] == 1:
                    draw.rectangle((0+krok*j, 0+krok*i, 2000/self.rozmiar + krok*j, 2000/self.rozmiar+krok*i), (0, 255, 0))

        return img

    # troche nie czuje sensu tych generatorow wiec pewnie zrobilem to bez sensu ale dziala xD
    def generator(self):
        index = 0
        magnetyzacja = 0

        while index < self.N:
            magnetyzacja = (np.sum(self.spiny)/(self.rozmiar*self.rozmiar))
            index += 1
            yield magnetyzacja, index

    def hamiltonian(self):
        H = 0
        kernel = (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        # wrap załatwia periodyczne warunki brzegowe B)
        self.convo = sp.convolve2d(self.spiny, kernel, mode='same', boundary='wrap')
        suma = np.sum(self.spiny * self.convo)

        # dziele przez 2 bo jest liczone podwojnie
        suma = suma/2
        H = - self.J * suma - self.B * np.sum(self.spiny)
        return H

    def krok(self):
        c = np.random.randint(0, self.rozmiar, size=(self.rozmiar**2))
        d = np.random.randint(0, self.rozmiar, size=(self.rozmiar**2))

        for i, j in zip(c, d):
            # licze od razu roznice a nie caly Hamiltionian
            zmiana_energii = (-2) * (-self.J * self.spiny[i, j] * self.convo[i, j] - self.B * self.spiny[i, j])
            # warunki na zaakceptowanie zmiany spinu
            if zmiana_energii <= 0:
                self.spiny[i, j] = -self.spiny[i, j]
            elif np.random.uniform(0.0, 1.0) < np.exp(-self.beta*zmiana_energii):
                self.spiny[i, j] = -self.spiny[i, j]



    def __call__(self):
        
        images = []
        if len(args.nazwa_pliku_z_magnetyzacja) != 0:
            outfile = open(f"{args.nazwa_pliku_z_magnetyzacja}.txt", 'w')

        for m, index in track(self.generator(), total=self.N):
            self.hamiltonian()
            #jesli podano prefix to zapisujemy png
            if self.prefix != []:   
                img = self.obrazek()
                img.save(f'{self.prefix}{index}.png')
            #jesli podano gif to zapisujemy
            if self.animacja != []:
                #jesli jednoczesnie nie podano prefixu to trzeba te rysunki i tak wygenerowac
                if self.prefix == []:
                    img = self.obrazek()
                images.append(img)
            #jesli podano plik na magnetyzacje to zapisujemy
            if len(args.nazwa_pliku_z_magnetyzacja) != 0:
                outfile.write(str(m) + '\n')
            
            self.krok()
        
        if self.animacja != []:
           # img = Image.new('RGB', (2000,2000), (0,0,0))
            images[0].save(f'{self.animacja}.gif', save_all=True, append_images = images[1:], duration = 300, loop=0)
        

s = Symulacja()
s()
