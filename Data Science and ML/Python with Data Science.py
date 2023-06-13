"""
Veri Bilimi Temellerini Öğrenmek İçin Ne Yapmak Gerekli ? 
1. Veri bilimi temel araçlarından bolan Python dilini ve veri bilimi ile ilgili Python kütüphanelerini nasıl kullanacağımızı öğrenmek.
2. Temel istatistik ve matematik kavramlarını bilmek.
3. Veri birleştirme / anlamlandırma, dağınık ve zor verilerle çalışma yöntemlerini kavramak.
4. Veri görselleştirme tekniklerine hakim olmak.
5. İleri seviye veri bilimi için makine öğrenmesi yöntemlerinin ardında yatan matematiği derinlemesine öğrenmek gerekse de temel seviyede bu yöntemleri sezgisel bir şekilde anlamak ve Python 
ile uygulamak.
6. Derin öğrenme yöntemleri veri biliminde ileri seviye tenikler olarak adlandırılabilir ama temel anlamda derin öğrenmenin ne olduğunu ve Python ile uygulamasını yapabilmek.

Veri Biliminnde Kullanılan Temel Kütüphaneler;
1. Pandas = Veri Bilimi Kütüphanesi
2. NumPy = Sayısal / İstatistik Kütüphanesi
3. Matplotlib = Veri Görselleştirme Kütüphanesi
4. SciPy = İstatistik Kütüphanesi
5. Sklearn = Makine Öğrenmesi Kütüphanesi
6. Keras = Derin Öğrenme Kütüphanesi

Veri Bilimi - Temel Kavramlar ve Tanımlar
Veri Bilimi (Veri Analitiiği)
Veri Madenciliği (Biçimlendirilmiş ve Belirli Bir Boyuta İndirgenmiş Veri)
Büyük Veri Analitiği (Biçimlendirilmemiş; Çok Miktarda Olan ve Hızlı Bir Şekilde Akamkta Olan Veri)

Veri Madenciliği - Tarihçe ve İsimlendirme
Veri Avlama: Veri Balıkçılığı (Data Fishing) - Veri Tarama (Data Dredging): 1960 - ....(İstatistikçiler)
Verir Tabanlarında Bilgi Keşfi (Knowledge Discovery in Databases - KDD): 1989 - .... (Yapay Zeka ve Makine Öğrenmesi Camialari)
Veri Madenciliği (Data Mining): 1990 - .... (Veritabanı Kullancıları)
Diğer İsimlendirmeler: Veri Arkeolojisi (Data Archaeology), Mahmut Hasadı (Information Harvesting), Malumat Keşfi (Information Discovery), Bilgi Çıkarımı (Knowledge Extraction)...
UYARI: Veri bilimi kursu bölümü boyunca, dersler birbirinin devamı niteliğindedir. Bu nedenle, yeni derse geçtiğiniz zaman bir önceki derslerde yazılan kodları koşturmanız gerekmektedir.
"""

#Kütüphanelerin import edilmesi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

#Python'da uyarıların kapatılması
import warnings
warnings.filterwarnings("ignore")

#Veriyi içeri aktarma (import etme)
veri = pd.read_csv("olimpiyatlar_temizlenmis_20220204.csv")
veri.head()

"""
Veri Hakkında Bilgi
Bu bölümde veri içerisinde bulunan her bir sütün içerisinde kaç adet örnek var, bu verilerin veri tipleri nedir  ve veri tiplerinin tüm veri içerisinde dağılımı nedir gibi veriyi tanımak 
için gerekli temel soruları cevaplayacağız.
"""
veri.info

"""
Verinin Temizlenmesi
Sütün İsimlerinin Düzenlenmesi
Yararsız Verinin Çıkarılması ve Düzenlenmesi
Kayıp Veri Sorunu

Sütun İsimlerinin Düzenlenmesi
Bu bölümde veri içerisinde bulunan sütun isimlerini inceleyeceğiz ve bu isimleri Türkçe'ye çevireceğiz.
"""

veri.columns #Datasetin sütün isimlerini verir.

#Sütun İsimlerini Değiştirme Fonksiyonu
veri.rename(columns = {'ID': 'id',
                       'Name': 'isim',
                       'Gender': 'cinsiyet',
                       'Age': 'yas',
                       'Height': 'boy',
                       'Weight': 'kilo',
                       'Team': 'takim',
                       'NOC': 'uok',
                       'Games': 'oyunlar',
                       'Year': 'yil',
                       'Season': 'sezon',
                       'City': 'sehir',
                       'Sport': 'spor',
                       'Event': 'etkinlik',
                       'Medal': 'madalya'}, inplace = True)
veri.head(2)
"""
Yararsız Verinin Çıkarılması ve Düzenlenmesi
ID sütunu veri hakkında herhangi bir bilgi vermediği için çıkarılacaktır.
Oyunlar sütunu yıl ve sezon sütunlarının birleşimi olduğu için yararsız veri olarak kabul edilir ve çıkarılabilir.
"""

#Drop Fonksiyonu ile ID ve Oyunlar Sütünlarını Çıkarlaım
veri = veri.drop(["id", "oyunlar"], axis = 1) #axis = 1 sütun, 0 ise satır manasında kullanılır.

"""
Kayıp Veri Sorunu
Kayıp veri kavramı veri içerisinde bulunan örneğin herhangi bir sütündan bulunan değerin olmaması anlamına gelir.
Mesela, aşağıda bulunan veride ikinci indexte bulunan örneğin boy ve kilo sütünlarından NaN (Not a Number) yazmaktadır.
Bu NaN yazısı örneğe ait boy ve kilo verisinin olmadığı ya da kayıp oludğu anlamına gelir.
"""

"""
Kayıp veri ile mücadele etmek için,
1. Kayıp veriye ait örnekler veriden çıkarılabilir.
2. Kayıt veriler asıl verinin dağılımını bozmayacak şekilde asıl verinin ortalama ya da medyan değerlerine göre doldurulabilir.
Bu veri üzerinden çalışma yaparken,
1. Boy ve kilo sütununda bulunan kayıt veriyi etkinlik ortalamasına göre dolduracağız.
2. Yaş sütunundan bulunan kayıp veriyi veri setinin yaş ortalamasına göre dolduracağız.
3. Madalya almayan sporcuları veri setinden çıkaracağız.
"""

"""
Boy ve Kilo Sütunu Kayıp Veri Doldurma
Boy ve kilo sütununda bulunan kayıp veriryi etkinlik ortalamasına göre dolduracağız.
"""

essiz_etkinlik = pd.unique(veri.etkinlik) 
print("Eşsiz Etkinlik Sayısı: {}".format(len(essiz_etkinlik)))
essiz_etkinlik[:10]

"""
Her bir etkinliği iteratif olarak dolaş.
Etkinlik özelinde boy ve kilo ortalamarını hesapla
Etkinlik özelinde kayıp boy ve kilo değerlerini etlinlik ortalamalarına eşitle.
"""

veri_gecici = veri.copy() #Gerçek veriyi bozmamak için bir kopyasını oluşturalım.
boy_kilo_liste = ["boy", "kilo"]

for e in essiz_etkinlik: #Liste içerisinde dolaş.

    #Etkinlik Filtresi Oluşturma
    etkinlik_filtre = veri_gecici.etkinlik == e
    #Veriyi Etkinliğe Göre Filtreleme
    veri_filtreli = veri_gecici[etkinlik_filtre]

    #Boy ve Kilo İçin Etkinlik Özelinde Ortamaları Hesaplayalım

    for s in boy_kilo_liste:
        ortalama = np.round(np.mean(veri_filtreli[s]), 2)
        if ~np.isnan(ortalama): #Eğer Etkinlik Özelinde Ortalama Varsa
            veri_filtreli[s] = veri_filtreli[s].fillna(ortalama)
        else: #Eğer Etlinlik Özelinde Ortalama Varrsa Ortalamayı Hesapla
            tum_veri_ortalamasi = np.round(np.mean(veri[s]), 2)
            veri_filtreli[s] = veri_filtreli[s].fillna(tum_veri_ortalamasi)
        #Etkinlik özelinde kayıp değerleri doldurulmuş olan veriyi, veri_gecici'ye eşitleyelim.
        veri_gecici[etkinlik_filtre] = veri_filtreli

#Kayıp değerleri giderilmiş olan geçici veriyi gerçek veriye eşitle.
veri = veri_gecici.copy()
veri.info() #Boy ve kilo sütunlarında kayıp değer sayısına bakalım.

"""
Yaş Sütunu Kayıp Veri Doldurma
Yaş sütununda bulunan kayıp veriyi veri setinin yaş ortalamasına göre dolduracağız.
"""

#Yaş değişkeninde tanımlı olmayan değerleri bulalım.

yas_ortalamasi =np.round(np.mean(veri.yas), 2)
print("Yaş Ortalaması: {}".format(yas_ortalamasi))
veri["yas"] = veri["yas"].fillna(yas_ortalamasi)
veri.info()

"""
Madalya Alamayan Sporcuları Veri Setinden Çıkar
Madalya alamayan sporcuları veri setinden çıkaracağız.
"""

madalya_degiskeni = veri["madalya"]
pd.isnull(madalya_degiskeni).sum()

madalya_degiskeni_filtreesi = ~pd.isnull(madalya_degiskeni)
veri.info()

#Sonradan kullanabilmek için temizlenmiş veriyi; veri.to_csv("olimpiyatlar_temizlenmis_csv", index = False)

"""
Tek Değişkenli Veri Analizi
Bu bölümde veri içerisinde bulunan sayısal ve kategorik verilerin tek tek incelemesini yapacağız.
Sayısal Değişkenler;
1. yas
2. boy
3. kilo
4. yil

Kategorik Değişkenler;
1. isim
2. cinsiyet
3. takim
4. UOK
5. sezon
6. sehir
7. spor 
8. etkinlik
9. madalya
"""

"""
Sayısal Değişkenler
Veri içerisinde 4 adet sayısal değişken bulunmaktadır. Bunlar;
1. yas
2. boy
3. kilo
4. yil
Sayısal değişkenleri incelemek ve yorumlamak için veri sıklığı ve te temel istatistik bilgilerini kullanacağız.
Veri sıklığı bilgilerini histogram gradiği sayesinde ineceleyeceğiz.
Temel istatistik bilgilerini ise kutu grafiği sayesinde yorumlayacağız.
"""

#Öncelikli olarak histogram grafiklerini çizdireceğimiz fonksiyonu yazalım.
def plotHist(degisken):
    """
Girdi: Değişken / sütun ismi
Çıktı: İlgili değişkenin histogramı
    """
    plt.figure()
    plt.hist(veri[degisken], bins = 85, color = "orange")
    plt.xlabel(degisken)
    plt.ylabel("Frekans")
    plt.title("Veri Sıklığı - {}".format(degisken))
    plt.show()

#Tüm sayısal edğişkenler için histogramları çizdirelim.

sayisal_degisken = ["yas", "boy", "kilo", "yil"]

for i in sayisal_degisken:
    plotHist(i)

"""
Beklenildiğği gibi yaş, kilo, ve boy sütunlarında normal dağılıma yakın bir dağılım söz konusu. (Normal Dağılımın özellikleri istatistik bölümünde daha detaylı ele alınacaktır.)
Yıl sütununda ise ilk olarak göze çarpan son yüzyılda gerçekleşen savaş zamanlarında olimpiyatların yapılmadığı ve son 20 yılda olimpiyatların eski yıllara göre daha çok yapıldığıdır.
"""

veri.describe() #Verilerin genel olarak matematiksel bilgierini (veri sayısı, ortalama, standart sapma, min, max ve %'lik değerlerini) verir.

"""
Sayısal Verinin İstatiksel Özellikleri;
1. count: Örnek Sayısı
2. mean: Ortalama
3. std: Standart Sapma
4. min: Minimum değer
5. 25%: Birinci Çeyrek
6. 50%: İkinci Çeyrek
7. 75%: Üçüncü Çeyrek
8. max: Maksimum Değer
"""

plt.boxplot(veri.yas)
plt.title("Yaş Değişkeni için Kutu Grafiği")
plt.xlabel("yas")
plt.ylabel("Değer")
plt.show()

"""
Kategorik Değişkenler
Veri içerisinde 9 adet kategorik değişken bulunmaktadır. Bunlar;
1. isim
2. cinsiyet
3. takim
4. uok
5. sezon
6. sehir
7. spor
8. etkinlik
9. madalya

Kategorik değişkenleri incelemek ve yorumlamak için bar grafiklerini kullanacağız.
"""

#Öncelikle bar grafiklerini çizdireceğimiz fonksiyonu yazalım.

def plotBar(degisken, n = 5):

        veri_ = veri[degisken] 
        veri_sayma = veri_.value_counts()
        veri_sayma = veri_sayma[:n]
        plt.figure()
        plt.bar(veri_sayma.index, veri_sayma, color = "orange")
        plt.xticks(veri_sayma.index, veri_sayma.index.values)
        plt.xticks(rotation = 45) #45 derecelik açı
        plt.ylabel("Frekans")
        plt.title("Veri Sıklığı - {}".format(degisken))
        plt.show()
        print("{}: \n {}".format(degisken, veri_sayma))

        kategorik_degisken = ["isim", "cinsiyet", "takim", "uok", "sezon", "sehir", "spor", "etkinlik", "madalya"]

        for i in kategorik_degisken:
            plotBar(i)

"""
Olimpiyat tarihi boyunca en çok madalya kazanan isim Micheal Phelps'dir.
Kadın ve erkek yarışmacılar arasında yaklaşık 1:2 oranı görülmektedir.
Takımlar olarak en çok ABD katılmıştır.
Beklendiği gibi UOK ve takım çıktıları aynıdır.
Yaz sezonunda düzenlenen olimğiyat sayısı kış sezonunda düzenlenenden fazladır.
Olimpiyatlar en çok Londra'da düzenlenmiştir.
Olimpiyatlarda en çok atletizm branşında yarışılmıştır.
Erkekler futbol müsabakaları en çok yapılan etkinliktir.
Verilen madalyaların aynı çıkmasını beklerken, farklılık olduğu görülmektedir.
"""

"""
İki Değişkenli Veri Analizi
İki değişkenli veri analizi yaparken kaznaılan madalyalar ve diğer sütünlar arasında bulunan ilişkileri inceleyeceğiz.
    1. Cinsiyete göre boy kilo karşılatırılması
    2. Sayısal sütunlar arasında ilişkinin incelenmesi
    3. Madalya ve yaş arasındaki ilişkinin incelenmesi
    4. Takımların kazandıkları altın, gümüş ve bronz madalya sayıları
    5. Kazanılan madalyaların hangi şehirlerde kazanıldığı
    6. Cinsiyete göre kazanılan altın, gümüş ve bron madalya sayıları
Görselleştirme bölümünde farklı görselleştirme teknikleri kullanılarak veri analizine devam edilecektir.
"""

#Cinsiyete Göre Boy ve Ağırlık Karşılaştırması

erkek = veri[veri.cinsiyet == "M"]
erkek.head(3)

kadin = veri[veri.cinsiyet == "F"]
kadin.head(3)

plt.figure()
plt.scatter(kadin.boy, kadin.kilo, alpha = 0.4, label = "Kadın", color = "pink")
plt.scatter(erkek.boy, erkek.kilo, alpha = 0.4, label = "Erkek", color = "blue")
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("Boy ve Kilo Arasındaki İlişki")
plt.legend()
plt.show()

#Boy ve ağırlık arasında pozitif ilişki olduğunu anlıyoruz.

#Sayısal Sütunlar Arasında İlişki İncelemesi

veri.loc[:,["yas", "boy", "kilo"]].corr() #Korelasyon Tablosu

#Bir önceki grafitke gördüğümüz boy ve ağırlık arasında pozitif ilişki durumunu, korelasyon tablosunda da görebiliyoruz.

#Madalya ve Yaş Arasındaki İlişki

#Sporcularımızı, altın, gümüş ve bronz madalyaya göre ayıralım.

veri_gecici = veri.copy()
veri_gecici = pd.get_dummies(veri_gecici, columns = ['madalya'])
veri_gecici.head(2)

veri_gecici.loc[:,["yas", "madalya_Bronze", "madalya_Gold", "madalya_Silver"]].corr()

#Takımların Kazandıkları Altın, Gümüş ve Bronz Madalya Sayıları

veri_gecici[["takim", "madalya_Bronze", "madalya_Gold", "madalya_Silver"]].groupby(["takim"], as_index = False).sum().sort_values(by = "madalya_Gold", ascending = False)[:10]

#Takımlara göre madalya çeşitlerinden en çok madalya kazanan takımlar ve madalya sayılarını çoktan aza şekilde ilk 10 takımı gösterecek şekilde sıralar.

#Kazanılan Madalyaların Hangi Şehirde Kazanıldığı

veri_gecici[["sehir", "madalya_Bronze", "madalya_Gold", "madalya_Silver"]].groupby(["sehir"], as_index = False).sum().sort_values(by = "madalya_Gold", ascending = False)[:10]

#Cinsiyete Göre Kazanılan Madalya Sayıları

veri_gecici[["cinsiyet", "madalya_Bronze", "madalya_Gold", "madalya_Silver"]].groupby(["cinsiyet"], as_index = False).sum().sort_values(by = "madalya_Gold", ascending = False)[:10]

"""
Çok Değişkenli Veri Analizi
Pivot Tablosu 
Pivot tablo, veritabanı, iş zekası programı ya da veri tablosu gibi, daha geniş bir tabloyu özetleyen istatistik tablosudur. Bu özet, pivot tabloyu anlamlı bir şekilde gruplayan toplamları, 
ortalamaları veya diğer istatistikleri içerebilir.
"""

veri_pivot = veri.pivot_table(index = "madalya", columns = "cinsiyet", values = ["boy", "kilo", "yas"], aggfunc = {"boy": np.mean, "kilo": np.mean, "yas": [min, max, np.std]})
#aggfunc = Aggregation Function ile birden fazla işlem tek bir satırda yapılabilir.
veri_pivot.head()

#Bu tabloda farklı madalyalar ve cinsiyetler için boy ve kilo parametrelerinin ortalama yaş parametresinin ise maksimum, minimum ve standart sapmasını görebiliyoruz.

"""
Anomali Tespiti
Aykırı değer, diğer gözlemelerden önemli ölçüde farklı bir veri noktasıdır. Bir aykırı değer, ölçümdeki değişkenliğe bağlı olabilir veya deneysel hatayı gösterebilir;
ikincisi bazen veri kümesinden çıkarılır. Bir aykırı değer istatiksel analizlerde ciddi sorunlara neden olabilir.
"""

def anomaliTespiti(df, ozellik):
    outlier_indices = []

    for c in ozellik:
        # 1. Çeyrek
        Q1 = np.percentile(df[c], 25) #Yüzdelik Dilim
        # 3. Çeyrek
        Q3 = np.percentile(df[c], 75)
        #IQR = Inter Quartile Range (Çeyrekler Açıklığı)
        IQR = Q3 - Q1
        # Aykırı değer için ek adım miktarı
        outlier_step = IQR * 1.5
        #Aykırı değeri ve de bulunduğu indexi tespit edelim.
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #Koşulun ilk kısmı negatif değerleri kapsarken ikinci kısmı pozitif değerleri kapsar.
        #Tespit edilen indeksleri depolayalım.
        outlier_indices.extend(outlier_list_col)

        #Eşsiz aykırı değerleri bulalım.
    outlier_indices = Counter(outlier_indices)
    #Eğer bir örnek v adet sütunda farklı değer ise bunu aykırı kabul edelim.
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1)

    return multiple_outliers
     
veri_anomali = veri.loc[anomaliTespiti(veri, ["yas", "kilo", "boy"])]
veri_anomali.spor.value_counts()


plt.figure()
plt.bar(veri_anomali.spor.value_counts().index, veri_anomali.spor.value_counts().values)
plt.xticks(rotation = 30)
plt.title("Anomaliye Rastlanan Spor Branşları: ")
plt.ylabel("Frekans")
plt.grid(True, alpha = 0.5)
plt.show()

veri_gym = veri_anomali[veri_anomali.spor == "Gymnastics"]
veri_gym 

veri_gym.etkinlik.value_counts()

"""
Jimnastikte meydana gelen anomali sonucunda veriyi incelediğimizde anomalinin nedeni sporcuların yaşlarının küçük ve kilolarının az olmasından kaynaklıdır.
Ayrıca jimnastikle meydana gelen anomalilerin hepsi kadın sporcular tarafından gerçekleşmiştir.
"""

veri_basketbol = veri_anomali[veri_anomali.spor == "Basketball"]
veri_basketbol

veri_basketbol.etkinlik.value_counts()

#Basketbol oynayan erkek oyuncularının boylarının uzun ve kilolarının fazla olmasından kaynaklı anomali tespit edilmiştir.

"""
Zaman Serilerinde Veri Analizi
    Yıllara Göre Ortalama, Yaş, Boy ve Kilo Değişimi
    Yıllara Göre Madalya Sarıları
"""

veri_zaman = veri.copy()
veri_zaman.head(3)

essiz_yillar = veri_zaman.yil.unique()
essiz_yillar

#Olimpiyatların Yapıldığı Yılların Sıralanması

yil_siralama = np.sort(veri_zaman.yil.unique())
yil_siralama

plt.figure()
plt.scatter(range(len(yil_siralama)), yil_siralama)
plt.grid(True)
plt.ylabel("Yıllar")
plt.title("Olimpiyatlar Çift Yıllarda Düzenlenir")
plt.show()

#Veri içerisinde bulunan yıl değerlerini datetime veri tipine dönüştürelim.

tarih_saat_nesnesi = pd.to_datetime(veri_zaman["yil"], format = '%Y')
print(type(tarih_saat_nesnesi))
tarih_saat_nesnesi.head(3)

veri_zaman["tarih_saat"] = tarih_saat_nesnesi #Veriye tarih_saat sütünu ekleme.
veri_zaman.head(3)

#veri_zaman değişkeninin ana indeksini, datetime tipi olan tarih_saat değerine göncelleyelim.

veri_zaman = veri_zaman.set_index("tarih_saat")
veri_zaman.drop(["yil"], axis = 1, inplace = True)
veri_zaman.head(3)

"""
Yıllara Göre Ortalama Yaş, Boy ve Kilo Değişimi
"""

periyodik_veri = veri_zaman.resample("2A").mean() #2 Yıllık periyotlar halinde ortalama değerleri alır.
periyodik_veri.head()

#Kayıp verileri çıkaralım.

periyodik_veri.dropna(axis = 0, inplace = True) #axis = 0 satırları, 1 ise sütunları belirtir.
periyodik_veri.head()

plt.figure()
periyodik_veri.plot()
plt.title("Yıllara Göre Ortalama Yaş, Boy ve Ağırlık Değişimi")
plt.xlabel("Yıl")
plt.grid(True)
plt.show()

#Yıllara Göre Madalya Sayıları

veri_zaman = pd.get_dummies(veri_zaman, columns = ['madalya'])
veri_zaman.head(3)

periyodik_veri = veri_zaman.resample("2A").sum() # 2 yıllık periyotlar halinde ortalama değerleri al
periyodik_veri.head()
# kayıp verileri çıkart
periyodik_veri = periyodik_veri[~(periyodik_veri == 0).any(axis=1)]
periyodik_veri.tail()

plt.figure()
periyodik_veri.loc[:,["madalya_Bronze", "madalya_Gold", "madalya_Silver"]].plot()
plt.title("Yıllara göre madalya sayıları")
plt.ylabel("sayı")
plt.xlabel("yıl")
plt.grid(True)
plt.show()

"""
Yukarıdaki şekilde görüldüğü gibi 2000'li yıllardan sonra madalya sayılarındaki dalgalanmalar fazla oluyor.
Bunun nedeni yaz ve kış olimpiyatlarında yapılan etkinlik sayısının ve buna karşılık gelen madalya sayısının farklılık göstermesidir.
"""

#Yıllara ve Sezonlara Göre Madalya Sayıları

yaz = veri_zaman[veri_zaman.sezon == "Summer"]
kis = veri_zaman[veri_zaman.sezon == "Winter"]
kis.head()

periyodik_veri_kis = kis.resample("A").sum()
periyodik_veri_kis = periyodik_veri_kis[~(periyodik_veri_kis == 0).any(axis = 1)]
periyodik_veri_kis.head()

periyodik_veri_yaz = yaz.resample("A").sum()
periyodik_veri_yaz = periyodik_veri_yaz[~(periyodik_veri_yaz == 0).any(axis = 1)]
periyodik_veri_yaz.head()

plt.figure()
periyodik_veri_yaz.loc[:, ["madalya_Bronze", "madalya_Gold", "madalya_Silver"]].plot()
plt.title("Yıllara Göre Madalya Sayıları - Yaz Sezonu")
plt.ylabel("Sayı")
plt.xlabel("Yıl")
plt.grid(True)
plt.show()

plt.figure()
periyodik_veri_kis.loc[:, ["madalya_Bronze", "madalya_Gold", "madalya_Silver"]].plot()
plt.title("Yıllara Göre Madalya Sayıları - Kış Sezonu")
plt.ylabel("Sayı")
plt.xlabel("Yıl")
plt.grid(True) 
plt.show()

