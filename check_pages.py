import sys
sys.path.insert(0, 'c:/Users/horie/chemai2')

pages = [
    ('data_load_page', 'frontend_streamlit.pages.data_load_page'),
    ('preprocess_page', 'frontend_streamlit.pages.preprocess_page'),
    ('eda_page', 'frontend_streamlit.pages.eda_page'),
    ('automl_page', 'frontend_streamlit.pages.automl_page'),
    ('evaluation_page', 'frontend_streamlit.pages.evaluation_page'),
    ('interpret_page', 'frontend_streamlit.pages.interpret_page'),
    ('dim_reduction_page', 'frontend_streamlit.pages.dim_reduction_page'),
    ('chem_page', 'frontend_streamlit.pages.chem_page'),
]

ok = 0
fail = 0
for name, mod in pages:
    try:
        __import__(mod)
        print('OK: ' + name)
        ok += 1
    except Exception as e:
        print('FAIL: ' + name + ' -- ' + str(e))
        fail += 1

print('')
print(str(ok) + '/' + str(ok+fail) + ' pages import OK')
