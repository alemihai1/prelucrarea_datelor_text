from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup
import urllib.request
import csv

filename = "pcgarage.csv"
with open(filename, 'w', newline='', encoding="utf-16") as csvfile:
    fieldnames = ['nume', 'product', 'rating', 'pro', 'contra', 'altele']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = "\t")
    writer.writeheader()

    req = urllib.request.Request("https://www.pcgarage.ro/sisteme/", headers={"User-Agent": "Chrome"})
    uclient = ureq(req)
    page_html2 = uclient.read()
    uclient.close()
    page_soup2 = soup(page_html2, "html.parser")

    subcategorii = page_soup2.findAll("small")

    for subcat in subcategorii:
        try:
            req = urllib.request.Request(subcat.a['href'], headers={"User-Agent": "Chrome"})
        except:
            continue
        uclient = ureq(req)
        page_html3 = uclient.read()
        uclient.close()
        page_soup3 = soup(page_html3, "html.parser")

        containers = page_soup3.findAll("div", {"class": "product_box_container"})
        ratingbox = page_soup3.findAll("div", {"class": "product_box_rating"})
        for container in ratingbox:
            try:
                req = urllib.request.Request(container.a['href'], headers={"User-Agent": "Chrome"})
            except:
                continue
            uclient = ureq(req)
            page_html4 = uclient.read()
            uclient.close()
            page_soup4 = soup(page_html4, "html.parser")

            comment_page = page_soup4.findAll("p", {"class": "ar_see_all clearfix"})
            req = urllib.request.Request(comment_page[0].a['href'], headers={"User-Agent": "Chrome"})
            uclient = ureq(req)
            page_html5 = uclient.read()
            uclient.close()

            page_soup5 = soup(page_html5, "html.parser")
            product_name = page_soup5.findAll("h1", {"id": "product_name"})
            print(product_name[0])
            reviews = page_soup5.findAll("div", {"itemprop": "review"})
            print(reviews[0])
            for review in reviews:
                row = {}
                name = review.findAll("span", {"itemprop": "author"})
                name = name[0].text.strip()
                rating = review.findAll("span", {"itemprop": "ratingValue"})
                rating = rating[0].text.strip()
                comments = review.findAll("p", {"itemprop": "description"})
                row['nume'] = name
                row['rating'] = rating
                row['product'] = product_name[0].span.text
                row['pro'] = "-"
                row['contra'] = "-"
                row['altele'] = "-"
                for comment in comments:
                    cat, *_ = comment.findAll("b")
                    if cat.text.strip() == 'Pro:' :
                        row['pro'] = (comment.text.strip()).replace("\n", " ").strip()
                    elif cat.text.strip() == 'Contra:' :
                        row['contra'] = (comment.text.strip()).replace("\n", " ").strip()
                    else:
                        row['altele'] = (comment.text.strip()).replace("\n", " ").strip()
                print(row)
                writer.writerow(row)
