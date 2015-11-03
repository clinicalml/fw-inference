rm -rf files.txt chinese*
ls ../ChineseCharTruncated/models/*.uai | sed 's/.uai//' | sed 's/..\/ChineseCharTruncated\/models\///' | sed 's/$/ 0 undir .\/config\/chinese_final.cfg/' | sed 's/^/python2.7 runExpt.py ChineseCharTruncated /' > files.txt
split -l 10 files.txt chinese
chmod +x chinese*
