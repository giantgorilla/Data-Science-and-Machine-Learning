"""
Veri Görselleştirme
    Veri Görselleştirme Nedir ?
        1. Bugün iş dünyasında veri analizi yoluyla toparlanan bu kadar çok bilgi ile bu bu verilerin bir resmini çizmenin bir yolunu bulmalıyız, böylece onları yorumlayabiliriz.
        2. Veri görselleştirme, haritalara veya grafiklere görsel bağşam sağlayarak bilginin ne anlama geldiğine dair bize net bir fikir verir.
        3. Bu, verileri insan zihninin anlaması için daha doğal hale getirir ve bu nedenle büyük veri kümelerindeki eğilimleri, modelleri ve aykırı değerleri tanımlamayı kolaylaştırır.

    Veri Görselleştirmenin Önemi
        1. Ana amaç, veri ve istatistikleri görselleştirme, bilgi edinmek için ekranları yorumlamaktır.
        2. Veri görselleştirme, veri temizleme, veri yapısını keşfetme, aykırı değerleri ve olağan dışı grupları teespit etme, eğilimleri ve kümeleri belirleme, yerel kalıpları belirleme, modelleme çıktısını değerlendirme ve sonuçları sunma için kullanışlıdır.
        3. Makine öğrenimi, tahmine dayalı analız gibi analizler yapmayı kolaylaştırır ve bu analizler, daha sonra sunum için yararlı görselleştirmeler işlevi görebilir.
        4. Finans, pazarlama, teknoloji, tasarım veya başka herhangi bir alanda çalışıyor olun, verileri görselleştirmeniz gerekir. Bu gerçek, veri görselleştirmenin önemini göstermektedir.

    Veri Görselleştirmede En Çok Kullanılan Python Kütüphaneleri
        1. Matplotlib: Python temellerini hatırlama bölümünde gördüğümüz temel görselleştirme kütüphanesidir.
        2. Seaborn: İstatiksel bir Python veri görselleştirme kütüphanesidir. 
        3. Plotly: Etkileşimli bir Python veri görselleştirme kütüphanesidir.
"""

#Kütüphanelerin Import Edilmesi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from collections import Counter

#Python'da Uyarıların Kapatılması
import warnings
warnings.filterwarnings("ignore")

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r+')
plt.xlabel("Tam Sayılar")
plt.ylabel("Tam Sayıların Kareleri")
plt.title("Bazı Tam Sayıların Kareleri")
plt.show()

t = np.arange(0.0, 5.0, 0.2)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^') #bs = Blue Square, g^ = Ters Üçgen
plt.show()

isimler = ['Ahmet', 'Mehmet', 'Ayşe', 'Fatma']
degerler = [5, 25, 50, 100]

plt.figure(figsize = (9,3))

plt.subplot(131)
plt.bar(isimler, degerler)

plt.subplot(132)
plt.scatter(isimler, degerler)

plt.subplot(133)
plt.plot(isimler, degerler)

plt.suptitle("Birçok Farklı Grafik") #SuperTitle - Grafiklerin Genel Başlığı
plt.show()

t = np.arange(0.0, 5.0, 0.2)
plt.plot(t, t, 'r-', linewidth = 2.0)
plt.show()


"""
mu1 = 100
sigma1 = 15
x = mu1 + (sigma1 * np.random.randn(10000))

mu2 = 110
sigma2 = 10
y = mu2 + (sigma2 * np.random.randn(10000))

#Söz konusu verilerin histogramı
n, bins, patches = plt.hist(x, 100, density = 1, facecolor = 'b', alpha = 0.25)
n, bins, patches = plt.hist(x, 100, density = 1, facecolor = 'g', alpha = 0.25)

plt.xlabel('Veriler')
plt.ylabel("Olasılıklar")
plt.title(r'Verilerin Histogramı: $\mu_1 = 100,\ \mu_2 = 110$')
plt.annotate("Gauss Eğrisinin Tepesi", xy = (110, 0, 0.04), xytext = (130, 0.042),
             arrowprops = dict (facecolor = 'black', shrink = 0.1))
plt.text(75, 0.025, r'$\mu_1 = 100,\ \sigma_1 = 15$')
plt.text(120, 0.035, r'$\mu_2 = 110,\ \sigma_2 = 10$')
plt.axis([40, 160, 0, 0.05])
plt.grid(True)
plt.show()
"""

x = np.arange(0, 10, 0.01)
y = np.exp(x) 

plt.figure()

plt.subplot(121) # 1 - 2 - 1 = 1. Düzlemde 2 grafikten 1.si
plt.plot(x, y)
plt.yscale("linear") #Doğrusal Fonksiyon
plt.title("Lineer (Doğrusal)")
plt.grid(True)

plt.subplot(122) # 1 - 2 - 2 = 1. Düzlemde 2 grafikten 2.si
plt.plot(x, y)
plt.yscale("log") #Logaritmik Fonksiyon
plt.title("Logaritmik")
plt.grid(True)

plt.show()

"""
Seaborn Kütüphanesi
Seaborn, Matplotlib kütüphanesi tabanlı, istatiksel bir Python veri görselleştirme kütüphanesidir.
Seaborn kullanıcılara istatiksel görselleştirmeler yapmaları için yüksek seviyeli bir arayüz sunar.
Seaborn kütüphanesinin işlevi;
    1.Birden çok değişken arasında ilişkileri detaylı bir şekilde incelemek adına sunulan veri kümesi tabanlı bir uygulama programlama arayüzüdür.
    2.Kategorik değişkenleri gözlemlemek ve üzerinde istatiksel işlemler yapmak için destek sağlar.
    3.Karmaşık görselleştirmeler oluşturmayı sağlayan çoklu grafikleri destekler.
    4.Alt kümeler arasında karşılaştırma yapmak için tek değişkenli ve çift değişkenleri görselleştirmeler sunar.
    5.Görselleştirmeyi daha anlaşılır hale getiren geniş renk skalası mevcuttur.
    6. Seaborn ve Matplotlib kütüphaneleri arasındaki farklar:
        6.1.Matplotlib kişiselleştirilebilir; fakat plotları daha anlaşılır, göze hitap eden şekle getirmek bir hayli zordur.
        6.2.Seaborn sahip olduğu renk paleti, arayüzler ve grafik çeşitleri ile kullanıcıya farklı çözümler sunar.
        6.3.Pandas kütüphanesi ile çalışırken Seaborn kolaylık sağlar.
"""

"""
120 Yıllık Olimpiyat Tarihi Veri Seti
"""

#Kütüphanelerin import edilmesi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from collections import Counter

#Python'da uyarıların kapatılması
import warnings
warnings.filterwarnings("ignore")

#Veriyi içeri aktarma (import etme)
veri = pd.read_csv("olimpiyatlar_temizlenmis.csv")
veri.head()

"""
Saçılım Grafiği
Saçılım grafiği, iki farklı değişkenin arasındaki ilişkiyi belirlemek için kullanılır.
"""

def sacilimGrafikCiz():
    sns.scatterplot(x = "boy", y = "kilo", data = veri) #X eksenine boy verilerini, Y eksenine kilo verilerini ve datayı çekeceği yol belirtilir.
    plt.title("Boy ve Ağırlık İlişkisi - Beyaz Izgara Tema")
    plt.show()

sacilimGrafikCiz()


sns.set_style("whitegrid")
sacilimGrafikCiz()

#Boy ve kilo arasında pozitif ilişki olduğu anlamını çıkarabiliriz.

#Saçılım Grafiği Kategorik

sns.scatterplot(x = "boy", y = "kilo", hue = "madalya", data = veri) #Hue = Madalyalara Göre Renklendirme
plt.title("Madalyaya Göre Boy ve Ağırlık Dağılımı")
plt.show()

sns.set_style("whitegrid")
sns.scatterplot(x = "boy", y = "kilo", hue = "cinsiyet", style = 'madalya', data = veri)
plt.xlabel("Sporcuların Boyları")
plt.ylabel("Sporcuların Ağırlığı")
plt.title("Madalyalı Sporcuların Ağırlık - Boy Değişimleri")
plt.show()

sns.set_style("darkgrid")

sns.scatterplot(x = "boy", y = "kilo", hue = "cinsiyet", style = 'madalya', size = 'yas', sizes = (15, 200), data = veri)
plt.xlabel("Sporcuların Boyları")
plt.ylabel("Sporcuların Ağırlığı")
plt.title("Madalyalı Sporcuların Ağırlık - Boy Değişimleri")
plt.show()

#Saçılım Grafiği - Doğrusal Regresyon

sns.regplot(x = "boy", y = "kilo", data = veri, marker = '+', scatter_kws= {'alpha' : 0.2})
plt.title("Boy ve Ağırlık Dağılımı")
plt.show()

#Doğrusal regresyon sonucu oluşan, doğrunun eğiminin pozitif olmasından anlaşılacağı üzere boy ve kilo arasında pozitif bir ilişki vardır.

#Renk Paletleri

sns.scatterplot(x = "boy", y = "kilo", hue = "madalya", data = veri, palette = 'Set1')
plt.title("Madalyaya Göre Boy ve Ağırlık Dağılımı")
plt.show()

"""
Çizgi Grafiği
    Çizgi grafiği, birbirine bağlanan düz çizgi parçalarının veri noktaları serisinin bilgisini gösteren bir tür çizimdir.
    Çizelege genellikle belirli bir zaman aralığındaki - zaman serisi - verilerin değerlerini görselleştirmek için kullanılır ve kronolojiiksel olarak çizilir.
    Ayrıca zamana bağlı ölçümlerin görselleştirilmesi için de kullanılır.
"""

sns.lineplot(x = "boy", y = "kilo", data = veri)
plt.title("Boy ve Kilo")
plt.show()

#Çizgi Grafiği - Kategorik

sns.lineplot(x = "boy", y = "kilo", hue = "cinsiyet", data = veri)
plt.title("Cinsiyete Göre Boy ve Ağırlık İlişkileri")
plt.show()

#Bu grafikten anlaşıldığı gibi kadınların boy ve kiloları erkeklere göre daha az olabiliyor.

"""
Histogram
    Histogram, gruplandırılmış b ir veri dağılımının sütün grafiğiyle gösterimidir.
    Tekrarlı sayılardan oluşan verilerin, uygulanan işlemlerden sonra önce tabloya, tabladan yararlanarak grafiğe aktarılması, yani veri grupalrının grafiğinin dikdörtgen sütünlar halinde gösterilmesidir. 
"""

sns.displot(veri, x = "kilo")
plt.ylabel("Frekans")
plt.title("Ağırlık Histogramı")
plt.show()

#Histogram - Kategorik

sns.displot(veri, x = "kilo", hue = "cinsiyet", legend = "lower")
plt.ylabel("Frekans")
plt.title("Cinsiyete Göre Ağırlık Histogramı")
plt.show()

sns.displot(veri, x = "kilo", col = "cinsiyet", multiple = "dodge") #Multiple = Cinsiyete Göre Ayrıştırarak 2 adet grafik oluşturur.
plt.show()

#Histogramlardan anlaşılabileceği üzere kadınların ortalama kilosu, erkeklerin ortalama kilosuna göre daha azdır.

#İki Boyutlu Histogram

sns.displot(veri, x = "kilo", y = "boy", kind = "kde")
plt.xlabel("Ağırlık")
plt.ylabel("Boy")
plt.title("Ağırlık - Boy Histogramı")
plt.show()

sns.displot(veri, x = "kilo", y = "boy", kind = "kde", hue = "cinsiyet")
plt.xlabel("Ağırlık")
plt.ylabel("Boy")
plt.title("Cinsiyete Göre Ayrıştırılmış Ağırlık - Boy Histogramı")
plt.show()

#Histogramdan anlaşılacağı üzere 2 adet peak noktası oluşmuş. Bu noktalardan birisi kadınlara aitken diğerleri erkeklere aittir.


sns.kdeplot(data = veri, x = "kilo", y = "boy", fill = True, hue = "cinsiyet") #Fill = İçlerinin dolu olup olmamasına karar verir.
plt.xlabel("Ağırlık")
plt.ylabel("Boy")
plt.title("Cinsiyete Göre Ayrıştırılmış Ağırlık - Boy Histogramı")
plt.show()

"""
Çubuk Grafiği
    Çubuk grafik veya çubuk grafik, kategorik verileri temsil ettikleri değerlerle orantılı yükseklik veya uzunluktaki dikdörtgen çubuklarla sunan bir grafiktir.
"""

sns.barplot(x = "madalya", y = "boy", data = veri)
plt.title("Madalyala Göre Boy Grafikleri")
plt.show()


sns.barplot(x = "madalya", y = "boy", data = veri, hue = "cinsiyet")
plt.title("Madalyala Göre Boy Grafikleri")
plt.show()

sns.catplot(x = "madalya", y = "yas", hue = "cinsiyet", col = "sezon", data = veri, kind = "bar", height = 4, aspect = 0.7) #Col = İstenilen sütüna göre o sütünda kaç adet değer var ise o kadar parçaya böler. 
plt.show()

"""
Kutu Grafiği
    İstatistik biliminde kutu grafiği bir betimsel istatistik ve istatiksel grafik aleti olup niceliksel verileri görsel şekilde özetlemek için kutu ve bıyıklar grafiği adı altında bir açıklayıcı veri analizi aracı olarak geliştirişmişit.
"""

sns.boxplot(x = "sezon", y = "boy", hue = "cinsiyet", data = veri, palette = "Set2")
plt.show()

veri_gecici = veri.loc[:, ["yas", "kilo", "boy"]]
sns.boxplot(data = veri_gecici, orient ="h", palette = "Set1") #H = Horizontal 
plt.show()

#Farklı madalyalar için yaz ve kış sezonlarında erkekler ve kadınlar arsında boy farklılıkları bu grafikten okunabiliyor.

"""
Isı Haritası
    Isı haritası, sayısal özelikler/sütünlar arasındaki ilişkiyi görselleştirmek iin kullanılan bir tekniktir.
"""

sns.heatmap(veri.corr(), annot = True, linewidths = 0.5, fmt = '.1f') #1f = Bir sayıyı yazarken virgülden sonra 1 basamak göster anlamına gelir. Annot = Renklerin içerisien değerleri yazdırır.
plt.show()

#Sayısal özellikler olan boy, kilo yaş ve yıllar arasında ilişkiyi incelemek için korelasyon matris kullandık.
#Daha önceden de bildiğimiz gibi boy ve kilo arasında 0.8 değerinde bir pozitif ilişki mevcuttur.

"""
Keman Grafiği
    Keman grafiği, sayısal verileri çizme yöntemidir.
    Kutu grafiğine benzer yorumlanır.
"""

sns.violinplot(x = "sezon", y = "boy", data = veri)
plt.show()


sns.violinplot(x = "sezon", y = "boy", data = veri, hue = "cinsiyet")
plt.show()


sns.violinplot(x = "sezon", y = "boy", data = veri, hue = "cinsiyet", split = True) #Split = Her birinin yarısını çizer.
plt.show()

sns.catplot(x = "sezon", y = "boy", hue = "cinsiyet", col = "madalya", data = veri, kind = "violin", split = True, height = 4, aspect = .7) 
plt.show()

"""
Ortaklık Grafiği
    Farklı özellikler arasındaki ilişkiyi farklı görselleştirme yöntemleri kullanarak incelemeye yarayan bir grafiktir.
"""

sns.jointplot(data = veri, x = "kilo", y = "boy", hue = "sezon", kind = "kde", height = 2, aspect = .5)
plt.show()

g = sns.JointGrid(data = veri, x = "kilo", y = "boy")
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot) #Kenarlara boxplot çizdirir.
plt.show()

"""
Çift/Eş Grafiği
    Veri seti içerisinde bulunan sayısal özlellikler arasındaki ilişkiyi farklı görselleştirme yöntemleri kullanarak incelemeye yaran bir grafiktir.
"""

sns.pairplot(hue = "sezon", data = veri)
plt.show()

g = sns.PairGrid(veri)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill = True)
g.map_diag(sns.histplot, kde = True) #diag = Diagonal

"""
Sayma Grafiği
    Sayma grafiği, niceliksel değişken yerine kategorik bir histogram olarak düşünülebilir.
"""

sns.countplot(x = "sehir", hue = "madalya", data = veri)
plt.xlabel("Şehir İsimleri")
plt.ylabel("Madalya Sayısı")
plt.legend(loc = "upper center")
plt.xticks(rotation = 90)
plt.show()

"""
Plotly Kütüphanesi
    Plotly Python programalama dili ile uyumlu çalışan veri görselleştirme ve dashboard oluşturmaya yarayan çeşitli fonksiyonlar içeren etkileşimli bir kütüphanedir.
    Plotly kullanılarak, interaktif grafikler birden fazla kişiyle çevrimici olarak kolayca paylaşılabilir.
    Plotly, verileri yükleyerek ve grafiksel kullanıcı arayüzü kullanarak etkileşimli grafikler oluşturmak için teknik geçmişi olmayan kişiler tarafından da kullanılabilir.
    Plotly ile etkileşimli grafikler oluşturmak için söz dizimi oldukça basittir.

    İçerik:
        1.Dünya Üniversite Sıralaması Veri Seti
        2.Çizgi Grafiği
        3.Saçılım Grafiği
        4.Çubuk Grafiği
        5.Dairesel Grafifk
        6.Kabarcık Grafiği
        7.Histogram
        8.Kutu Grafiği
        9.Saçılım Matrisi Grafiği
        10.Alt Plan Grafiği
        11.3D Saçılım Grafiği
        12.Dünya Haritası Grafiği
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as plt
init_notebook_mode(connected = True)
import plotly.graph_objects as go #Graph Objects
import plotly.io as pio #Plotly IO
pio.renderers.default = "vscode"
import plotly.figure_factory as ff #Figure Factory
import plotly.express as px


veri = pd.read_csv("universite_siralamasi_20220204.csv")
veri.head(3)

veri.columns

veri.rename(columns = {
    "world_rank": "dunya_siralama",
    "university_name": "universite_adi",
    "country": "ulke",
    "teaching": "ogretim",
    "international": "uluslararasi",
    "research": "arastirma",
    "citations": "alinti",
    "income": "gelir",
    "total_score": "toplam puan",
    "num_students": "ogrenci_sayisi",
    "student_staff_ratio": "ogrenci_calisan_orani",
    "international_student": "uluslararası_ogrenci",
    "female_male_ratio": "kadin_erkek_orani",
    "year": "yil"    
    }, inplace = True)

veri.head(3)

"""
Çizgi Grafiği
    Çizgi grafiği, birbirine bağlanan düz çizgi parçalarının veri noktaları serisinin bilgisini gösteren bir tür çizimdir.
    Çizelge genellikle belirli bir zaman aralığındaki - zaman serisi - verilerin değerlerini görselleştirmek için kullanılır ve kronolojiksel olarak çizilir.
    Ayrıca zamana bağlı ölçümlerin görselleştirilmesi için de kullanılır.
"""

df = veri.iloc[:100, :]
df.head(3)

#Plotly Stilleri

pio.templates.default = "simple_white" #BackGround

cizgi1 = go.Scatter(x = veri.dunya_siralama, 
                    y = veri.alinti, 
                    mode = "markers", 
                    name = "Alıntı", 
                    marker = dict(color = 'rgba(78, 78, 250, .85)'), 
                    text = veri.universite_adi) #Hover = Üzerinde gezinirken üniversite adının gösterilmesi.

cizgi2 = go.Scatter(x = veri.dunya_siralama, 
                    y = veri.ogretim, 
                    mode = "markers", 
                    name = "Eğitim - Öğretim Faaliyetleri", 
                    marker = dict(color = 'rgba(202, 43, 15, .85)'), 
                    text = veri.universite_adi) #Hover = Üzerinde gezinirken üniversite adının gösterilmesi.

veri = [cizgi1, cizgi2]
yerlesim = dict(title = "Dünya Sıralamasında İlk 100 Üniversitenin Atıf ve Eğitim - Öğretim Puanları",
                xaxis = dict(title = "Dünya Sıralaması", 
                ticklen = 5, #Tik Uzunluğu
                zeroline = False))

fig = dict(data = veri,
           layout = yerlesim)
plot(fig, filename = "1_cizgi_alinti ve öğretim puanları.html")

veri2011 = df[veri.yil == 2011].iloc[:5, :]

cizgi1 = go.Bar(x = veri2011.universite_adi, 
                y = veri2011.alinti, 
                name = "Alıntı", 
                marker = dict(color = 'rgba(255, 127, 40, .5)'),
                base = dict(color = 'rgb(0, 0, 0)', width = 1.5), 
                text = veri2011.ulke) #Hover = Üzerinde gezinirken üniversite adının gösterilmesi.

cizgi2 = go.Bar(x = veri2011.universite_adi, 
                y = veri2011.ogretim,   
                name = "Eğitim Öğretim Faaliyetleri", 
                marker = dict(color = 'rgba(64, 10, 128, .5)'),
                base = dict(color = 'rgb(0, 0, 0)', width = 1.5), 
                text = veri2011.ulke) #Hover = Üzerinde gezinirken üniversite adının gösterilmesi.

veri_ = [cizgi1, cizgi2]
yerlesim = go.Layout(barmode = "group")
fig = go.Figure(data = veri_,
                layout = yerlesim)

plot(fig, filename = "2_cubuk_alinti ve öğretim puanları.html")


veri2011 = df[df.yil == 2011].iloc[: 8, :]
dilim1 = veri2011.ogrenci_sayisi
dilim1_liste = [float(each.replace(',', '.')) for each in veri2011.ogrenci_sayisi]
etiketler = veri2011.universite_adi

cizgi = go.Pie(labels = etiketler,
               values = dilim1_liste,
               hoverinfo = "label+value+percent",
               textinfo = "value+percent",
               textfont = dict(size = 8),
               rotation = 100,
               hole = 0.3,
               marker = dict(line = dict (color = 'seagreen', width = 1)))

veri_ = [cizgi]
yerlesim = dict(title = "2021 Yılı - İlk 8 Üniversitenin Öğrenci Sayıları ve Sıralamaları",
                legend = dict(orientation = "h"))
fig = dict(data = veri_, layout = yerlesim)
plot(fig, filename = "3_dairesel ogrenci sayisi.html")

"""
Kabarcık Grafiği
    Saçılım grafiği olarak da adlandırılabilir.
    Saçılım noktalarına verilen renk sayesinde görselleştirmeye yeni bir boyut katar.
"""

"""
veri2011 = df[df.yil == 2011].iloc[: 8, :]
ogrenci_sayisi = [float(each.replace(',', '.')) for each in veri2011['ogrenci_sayisi']]
uluslararası_renk = [float(each.replace(',', '.')) for each in veri2011['uluslararası_ogrenci']]

cizgi = go.Scatter(x = "ogretim",
                   y = "dunya_siralama",
                   marker = dict(color = "seagreen"),
                   text = veri2011.universite_adi)

veri_ = [cizgi]
yerlesim = dict(title = "XYZ")

fig = dict(data = veri_,
           layout = yerlesim)

plot(fig, filename = "4_kabarcık.html")
"""

