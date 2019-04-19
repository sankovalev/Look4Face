from django.shortcuts import render, redirect
import logging
logging.basicConfig(filename="look4face.log", level=logging.INFO)

def main(request):
    """Displays the main page
    
    Arguments:
        request {[type]} -- [description]
    """
    logger = logging.getLogger('main')
    if request.method == 'GET':
        # try:
        context = {
            'title': 'Главная страница',
            }
        return render(request, 'index.html', context)
        # except Exception as e:
        #     logger.error(f'GET-request, {str(e)}')
        #     return redirect('Main Page')
    else:
        logger.warning(f'Запрос не обработан, {str(e)}')
        return redirect('Main Page')
