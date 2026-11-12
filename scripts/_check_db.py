import pymysql
from dotenv import load_dotenv
load_dotenv()

try:
    conn = pymysql.connect(host='localhost', user='root', password='', database='kawaltani_rag_tesis',
                           cursorclass=pymysql.cursors.DictCursor)
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) as n FROM tm_sensor_read')
    print('kawaltani_rag_tesis.tm_sensor_read rows:', cur.fetchone()['n'])
    conn.close()
except Exception as e:
    print('kawaltani_rag_tesis ERROR:', e)

try:
    conn2 = pymysql.connect(host='localhost', user='root', password='', database='kawaltani_simpul_telkom',
                            cursorclass=pymysql.cursors.DictCursor)
    cur2 = conn2.cursor()
    cur2.execute('SHOW TABLES')
    tables = [list(r.values())[0] for r in cur2.fetchall()]
    print('\nkawaltani_simpul_telkom tables:')
    for t in tables:
        cur2.execute('SELECT COUNT(*) as n FROM `{}`'.format(t))
        print('  {}: {} rows'.format(t, cur2.fetchone()['n']))
    conn2.close()
except Exception as e:
    print('kawaltani_simpul_telkom ERROR:', e)

